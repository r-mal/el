from abc import ABC

import tensorflow as tf
import numpy as np
import os
from hedgedog.tf.typing import TensorDict, TensorOrTensorDict
from hedgedog.tf import layers as hdlayers
from hedgedog.tf import metrics as hdmetrics
from hedgedog.logging import get_logger
import hedgedog.tf.models.bert as modeling

from el.model.normalization import NormalizationModule
from el.config import model_ing

log = get_logger("el.model.cake")


class CakeModule(NormalizationModule):
  @model_ing.capture
  def __init__(self, params, is_training, ace_path: str, train_bert: bool, include_cls_sep: bool = True):
    super().__init__(params, is_training)
    # half, just embs, no proj_embs
    self.embedding_size = 50
    self.rnn_num_layers = 1
    self.rnn_hidden_size = 512
    self.train_bert = train_bert
    self.include_cls_sep = include_cls_sep
    self.latest_model_checkpoint = tf.train.latest_checkpoint(ace_path)

    log.info(f"Initialized CAKE module.")

  def __call__(self, shared_representation: TensorOrTensorDict, features: TensorDict) -> TensorDict:
    # [b, n, d]
    contextualized_embeddings = shared_representation['contextualized_tokens']
    if not self.train_bert:
      contextualized_embeddings = tf.stop_gradient(contextualized_embeddings)
    s_lens = shared_representation['slens']

    b, n, d = hdlayers.get_shape_list(contextualized_embeddings)
    # [b, c, n]
    concept_token_masks = features['concept_token_masks']
    _, c, _ = hdlayers.get_shape_list(concept_token_masks)

    # [b, c]
    concept_head_idxs = features['concept_head_idx']
    # [b, c]
    concept_lengths = tf.cast(tf.reduce_sum(concept_token_masks, axis=-1), tf.int32)

    # [bsize, max_num_concepts, max_concept_length, emb_size]
    concept_context_embeddings, concept_lengths = concept_index(
      contextualized_embeddings,
      concept_head_idxs,
      concept_lengths,
      s_lens,
      include_cls_sep=self.include_cls_sep
    )
    max_concept_length = tf.cast(tf.reduce_max(concept_lengths), tf.int32)
    # [bsize, max_num_concepts, max_concept_length, 1]
    concept_seq_mask = tf.expand_dims(tf.sequence_mask(concept_lengths, max_concept_length, dtype=tf.float32), axis=-1)

    # mask concept token embeddings by concept length mask
    concept_context_embeddings = concept_context_embeddings * concept_seq_mask

    with tf.variable_scope('ace_encoder'):
      with tf.variable_scope('rnn_concept_encoder', reuse=tf.AUTO_REUSE):
        # [b, c, 1024]
        concept_encodings = self._concept_rnn(concept_context_embeddings, concept_lengths)

    with tf.variable_scope('transd_embeddings'):
      with tf.variable_scope('concept_embeddings', reuse=tf.AUTO_REUSE):
        # [b, c, 50]
        concept_embeddings = tf.layers.dense(
          inputs=concept_encodings,
          units=self.embedding_size,
          activation=None,
          name='embeddings'
        )

    log.info('Initializing CAKE from checkpoint...')
    init_from_checkpoint(self.latest_model_checkpoint)
    log.info('Initialized CAKE from checkpoint.')

    # https://www.aclweb.org/anthology/P15-1067.pdf
    # normalize all lookups
    # removed in v51+
    # embeddings_norm = tf.norm(concept_embeddings, ord=2, axis=-1, keepdims=True)
    # concept_embeddings = concept_embeddings / tf.maximum(embeddings_norm, 1.0)
    mention_embeddings = tf.reshape(
      concept_embeddings,
      shape=[b, c, self.embedding_size]
    )
    rnn_pooled = tf.reshape(
      concept_encodings,
      shape=[b, c, concept_encodings.get_shape()[-1]]
    )

    features['pooled_mention'] = rnn_pooled
    features['mention_embeddings'] = mention_embeddings
    return features

  def _concept_rnn(self, concept_embeddings, concept_lengths):
    concept_lengths = tf.cast(concept_lengths, tf.int64)
    bsize = tf.shape(concept_embeddings)[0]
    max_concept_length = tf.cast(tf.reduce_max(concept_lengths), tf.int32)
    max_num_concepts = tf.shape(concept_embeddings)[1]
    context_emb_size = concept_embeddings.get_shape()[-1]
    # [bsize * max_num_concepts, max_concept_length, context_emb_size]
    concept_embeddings_flat = tf.reshape(
      concept_embeddings,
      shape=[bsize * max_num_concepts, max_concept_length, context_emb_size]
    )
    # [bsize * max_num_concepts]
    concept_lengths_flat = tf.reshape(
      concept_lengths,
      shape=[bsize * max_num_concepts]
    )

    # [bsize * max_num_concepts, max_concept_length, context_emb_size]
    # [bsize * max_num_concepts]
    concept_embeddings_flat, concept_lengths_flat = tf.cond(
      max_concept_length > 0,
      true_fn=lambda: (concept_embeddings_flat, concept_lengths_flat),
      # [bsize*1, 1, context_emb_size]
      # Throw-away masked out later in case of batch which contains no concepts.
      false_fn=lambda: (tf.zeros(shape=[bsize, 1, context_emb_size], dtype=tf.float32),
                        # [bsize*1]
                        tf.ones(shape=bsize, dtype=tf.int64))
    )

    # padded concepts will have a length of 0 so give them a fake length of 1, will be masked out later.
    concept_lengths_flat = tf.where(
      concept_lengths_flat == 0,
      tf.ones_like(concept_lengths_flat),
      concept_lengths_flat
    )

    # [bsize * max_num_concepts, rnn_emb_size]
    concept_head_embeddings = rnn_encoder(
      concept_embeddings_flat,
      concept_lengths_flat,
      nrof_layers=self.rnn_num_layers,
      nrof_units=self.rnn_hidden_size,
      rnn_type='lstm',
      reuse=tf.AUTO_REUSE
    )

    # [bsize, max_num_concepts, rnn_emb_size]
    concept_head_embeddings = tf.cond(
      max_concept_length > 0,
      true_fn=lambda: tf.reshape(
        concept_head_embeddings,
        shape=[bsize, max_num_concepts, 2 * self.rnn_hidden_size]
      ),
      # if the max_concept_length was 0 then we ignore rnn entirely
      false_fn=lambda: tf.zeros(shape=[bsize, 0, 2 * self.rnn_hidden_size], dtype=tf.float32)
    )
    return concept_head_embeddings


def concept_index(contextualized_embeddings, concept_head_idxs, concept_lengths, s_lens, include_cls_sep=True):
  max_concept_length = tf.cast(tf.reduce_max(concept_lengths), tf.int32)
  # [1, 1, max_concept_length]
  concept_range = tf.expand_dims(tf.expand_dims(tf.range(max_concept_length, dtype=tf.int32), axis=0), axis=0)
  # [bsize, max_num_concepts, 1]
  # [bsize, max_num_concepts, max_concept_length]
  concept_idxs = tf.expand_dims(tf.cast(concept_head_idxs, tf.int32), axis=-1) + concept_range
  b, c, l = hdlayers.get_shape_list(concept_idxs)

  # ensure valid indices for indices outside range (will be ignored by rnn due to lengths)
  # this can occur because max_concept_length + max concept start idx can overflow past
  # end of sequence, so just max at end and rnn will ignore using length info
  # [bsize, c, l]
  s_len_pad = tf.tile(
    tf.expand_dims(tf.expand_dims(s_lens - 1, axis=-1), axis=-1),
    multiples=[1, c, l]
  )
  concept_idxs = tf.maximum(tf.minimum(concept_idxs, s_len_pad), 0)

  if include_cls_sep:
    # [b, c, 1]
    cls_token_idxs = tf.zeros(shape=[b, c, 1], dtype=tf.int32)
    # [b, c, 1]
    sep_token_idxs = tf.tile(
      tf.expand_dims(tf.expand_dims(s_lens - 1, axis=-1), axis=-1),
      multiples=[1, c, 1]
    )
    # [b, c, l+2]
    concept_idxs = tf.concat(
      [
        cls_token_idxs,
        concept_idxs,
        sep_token_idxs
       ],
      axis=-1
    )
    concept_lengths = concept_lengths + 2

  # [bsize, max_num_concepts, max_concept_length, emb_size]
  concept_embeddings = tf.gather(
    # [bsize, num_tokens, emb_size]
    contextualized_embeddings,
    # [bsize, max_num_concepts, max_concept_length]
    concept_idxs,
    batch_dims=1
  )
  return concept_embeddings, concept_lengths


def rnn_encoder(input_embs, input_lengths, nrof_layers, nrof_units, rnn_type, reuse=tf.AUTO_REUSE):
  seq_output_indices = input_lengths - 1
  if rnn_type == 'gru':
    cell = tf.contrib.cudnn_rnn.CudnnGRU
  elif rnn_type == 'lstm':
    cell = tf.contrib.cudnn_rnn.CudnnLSTM
  else:
    raise ValueError(f'Unknown rnn type: {rnn_type}')
  with tf.variable_scope('forward', reuse=reuse) as scope:
    rnn_forward = cell(
      num_layers=nrof_layers,
      num_units=nrof_units
    )
    input_embs_seq_major = tf.transpose(input_embs, [1, 0, 2])
    encoder_forward_seq_major, _ = rnn_forward(input_embs_seq_major, scope=scope)
    encoder_forward = tf.transpose(encoder_forward_seq_major, [1, 0, 2])
    encoder_forward = extract_last_seq_axis(encoder_forward, seq_output_indices)

  with tf.variable_scope('backward', reuse=reuse) as scope:
    rnn_backward = cell(
      num_layers=nrof_layers,
      num_units=nrof_units
    )
    input_embs_rev = tf.reverse_sequence(
      input_embs,
      input_lengths,
      seq_axis=1,
      batch_axis=0)
    input_embs_seq_major_rev = tf.transpose(input_embs_rev, [1, 0, 2])
    encoder_backward_seq_major_rev, _ = rnn_backward(input_embs_seq_major_rev, scope=scope)
    encoder_backward = tf.transpose(encoder_backward_seq_major_rev, [1, 0, 2])
    encoder_backward = extract_last_seq_axis(encoder_backward, seq_output_indices)

  encoder_out = tf.concat([encoder_forward, encoder_backward], axis=1, name='encoder_out')
  return encoder_out


def extract_last_seq_axis(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.cast(tf.range(tf.shape(data)[0]), tf.int64)
    indices = tf.stack([batch_range, tf.cast(ind, tf.int64)], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def init_from_checkpoint(init_checkpoint):
  t_vars = tf.trainable_variables()
  (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
    t_vars,
    init_checkpoint
  )
  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  for trainable_var in t_vars:
    if trainable_var.name in initialized_variable_names:
      init_string = '*INIT_FROM_CKPT*'
      print(f'{trainable_var.name}: {trainable_var.get_shape()} {init_string}')
