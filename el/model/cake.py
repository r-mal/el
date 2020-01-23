from abc import ABC

import tensorflow as tf
import numpy as np
import os
from hedgedog.tf.estimator.multitask import Module
from hedgedog.tf.typing import TensorDict, TensorOrTensorDict
from hedgedog.tf import layers as hdlayers
from hedgedog.tf import metrics as hdmetrics
from hedgedog.logging import get_logger
import hedgedog.tf.models.bert as modeling

from el.config import model_ing

log = get_logger("el.model.cake")


class CakeModule(Module):
  @model_ing.capture
  def __init__(self, params, is_training, umls_embeddings: str, ace_path: str, cake_loss_fn: str, cake_margin: float):
    super().__init__(params, is_training)
    self.code_embeddings = self._init_embeddings(params, umls_embeddings)
    # half, just embs, no proj_embs
    self.embedding_size = 50
    self.rnn_num_layers = 1
    self.rnn_hidden_size = 512
    self.latest_model_checkpoint = tf.train.latest_checkpoint(ace_path)
    self.cake_loss_fn = cake_loss_fn
    self.cake_margin = cake_margin

    log.info(f"Initialized CAKE module.")

  # noinspection PyMethodMayBeStatic
  def _init_embeddings(self, params, umls_embeddings):
    import json
    with np.load(os.path.join(params.dataset.project_dir, 'info', umls_embeddings, 'embeddings.npz')) as npz:
      np_code_embeddings = npz['embs']
    code_embeddings = tf.Variable(np_code_embeddings, trainable=False, name='umls_embeddings')

    code2id = json.load(open(os.path.join(params.dataset.project_dir, 'info', 'cui2id.json')))
    num_learnable_codes = len(code2id) - np_code_embeddings.shape[0]

    # pop a quick 'NO CUI' embedding on the matrix
    mean = np.mean(np_code_embeddings)
    std = np.std(np_code_embeddings)
    init = tf.initializers.random_normal(mean=mean, stddev=std, seed=1337)
    no_cui_emb = tf.get_variable('learnable_embeddings',
                                 shape=[num_learnable_codes, np_code_embeddings.shape[-1]],
                                 initializer=init,
                                 trainable=True)
    return tf.concat((code_embeddings, no_cui_emb), axis=0)

  def __call__(self, shared_representation: TensorOrTensorDict, features: TensorDict) -> TensorDict:
    # [b, n, d]
    contextualized_embeddings = shared_representation['contextualized_tokens']
    # s_lens = shared_representation['slens']

    b, n, d = hdlayers.get_shape_list(contextualized_embeddings)
    # [b, c, n]
    concept_token_masks = features['concept_token_masks']
    _, c, _ = hdlayers.get_shape_list(concept_token_masks)

    # [b, c]
    concept_head_idxs = features['concept_head_idx']
    # [b, c]
    concept_lengths = tf.cast(tf.reduce_sum(concept_token_masks, axis=-1), tf.int32)

    max_concept_length = tf.cast(tf.reduce_max(concept_lengths), tf.int32)
    # [bsize, max_num_concepts, max_concept_length, 1]
    concept_seq_mask = tf.expand_dims(tf.sequence_mask(concept_lengths, max_concept_length, dtype=tf.float32), axis=-1)
    # [bsize, max_num_concepts, max_concept_length, emb_size]
    concept_context_embeddings = self._concept_index(
      contextualized_embeddings,
      concept_head_idxs,
      concept_lengths
    )

    # mask concept token embeddings by concept length mask
    concept_context_embeddings = concept_context_embeddings * concept_seq_mask

    with tf.variable_scope('ace_encoder'):
      with tf.variable_scope('rnn_concept_encoder', reuse=tf.AUTO_REUSE):
        # [b, c, 1024]
        concept_encodings = self._crnn(concept_context_embeddings, concept_lengths)

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
    embeddings_norm = tf.norm(concept_embeddings, ord=2, axis=-1, keepdims=True)
    concept_embeddings = concept_embeddings / tf.maximum(embeddings_norm, 1.0)
    mention_embeddings = tf.reshape(
      concept_embeddings,
      shape=[b, c, self.embedding_size]
    )
    rnn_pooled = tf.reshape(
      concept_encodings,
      shape=[b, c, tf.shape(concept_encodings)[-1]]
    )

    features['pooled_mention'] = rnn_pooled
    features['mention_embeddings'] = mention_embeddings
    return features

  def predict(self, graph_outputs_dict: TensorDict) -> TensorDict:
    if 'candidates' in graph_outputs_dict:
      scores, _ = self._calc_scores(graph_outputs_dict, graph_outputs_dict)
      graph_outputs_dict['candidate_scores'] = scores

    # graph_outputs_dict.__delitem__('embedding_weight')
    return graph_outputs_dict

  def loss(self, graph_outputs_dict: TensorDict, labels: TensorDict) -> TensorOrTensorDict:
    scores, gold_negative_scores = self._calc_scores(graph_outputs_dict, labels)

    pos_score = scores[:, :, 0]
    negative_scores = scores[:, :, 1:]

    # [b, c]
    losses = self.loss_fn(pos_score, negative_scores, gold_negative_scores)
    concept_mask = tf.sequence_mask(graph_outputs_dict['num_concepts'], dtype=tf.float32)
    loss = tf.reduce_mean(losses * concept_mask)

    return {"normalization_loss": loss}

  def loss_fn(self, pos_score, negative_scores, gold_negative_scores):
    # -1 * -1 * energy = energy
    # TODO consider using negative scores in the future
    if self.cake_loss_fn == 'energy':
      loss = -pos_score
    elif self.cake_loss_fn == 'margin':
      pos_score = tf.expand_dims(pos_score, axis=-1)
      loss = tf.nn.relu(self.cake_margin - pos_score + negative_scores)
    elif self.cake_loss_fn == 'relative_margin':
      # closer to gold than gold is close to other candidates
      # [b, c, k-1]
      # [b, c, 1]
      pos_score = tf.expand_dims(pos_score, axis=-1)
      loss = tf.reduce_mean(
        tf.nn.relu(
          gold_negative_scores - pos_score + negative_scores
        ),
        axis=-1
      )
    else:
      raise ValueError(f'Unknown cake loss function: {self.cake_loss_fn}')
    return loss

  def eval_metrics(self, graph_outputs_dict: TensorDict, labels: TensorDict, loss: TensorOrTensorDict) -> TensorDict:
    # [b, c, k+1]
    scores, _ = self._calc_scores(graph_outputs_dict, labels)

    # [b, c]
    ones = tf.sequence_mask(graph_outputs_dict['num_concepts'], dtype=tf.bool)
    maximum_negative_score = tf.reduce_max(scores[:, :, 1:], axis=-1)
    # correct_predictions = tf.greater(pos_score, maximum_negative_score)
    max_idx = tf.argmax(scores, axis=-1)
    correct_predictions = tf.equal(max_idx, 0)
    strict_predictions = tf.logical_and(correct_predictions, tf.cast(labels['gold_in_candidate_mask'], tf.bool))

    # eval all data
    accuracy = tf.metrics.accuracy(labels=ones,
                                   predictions=correct_predictions,
                                   weights=ones)
    num_correct = tf.metrics.true_positives(labels=ones,
                                            predictions=correct_predictions,
                                            weights=ones)
    total = hdmetrics.count(ones)

    # strict: assume concepts without gold cui in candidate set is incorrect
    strict_accuracy = tf.metrics.accuracy(labels=ones,
                                          predictions=strict_predictions,
                                          weights=ones)
    strict_correct = tf.metrics.true_positives(labels=ones,
                                               predictions=strict_predictions,
                                               weights=ones)
    gold_in_candidates = hdmetrics.count(tf.cast(labels['gold_in_candidate_mask'], tf.bool))

    # normalized: ignore concepts without gold cui in candidate set
    normalized_accuracy = tf.metrics.accuracy(labels=ones,
                                              predictions=correct_predictions,
                                              weights=labels['gold_in_candidate_mask'])

    eval_metrics = {
      self.params.model.norm_loss_fn: tf.metrics.mean(loss['normalization_loss']),
      'normalization/accuracy': accuracy,
      'normalization/normalized_accuracy': normalized_accuracy,
      'normalization/strict_accuracy': strict_accuracy,
      'normalization/avg_pos_score': tf.metrics.mean(scores[:, :, 0]),
      'normalization/avg_neg_score': tf.metrics.mean(scores[:, :, 1:]),
      'normalization/avg_max_neg_score': tf.metrics.mean(maximum_negative_score),
    }

    if self.params.model.verbose_eval:
      eval_metrics['total'] = total
      eval_metrics['gold_in_candidates'] = gold_in_candidates
      eval_metrics['num_correct'] = num_correct
      eval_metrics['num_correct_strict'] = strict_correct
    return eval_metrics

  def _calc_scores(self, graph_outputs_dict, labels):
    # [b, c, dim]
    mention_embeddings = graph_outputs_dict['mention_embeddings']
    # [b, c, k]
    candidates = labels['candidates']
    # [b, c, k]
    candidate_mask = labels['candidate_mask']
    # [b, c, k, dim] where c is number of candidates
    candidate_embeddings = tf.nn.embedding_lookup(self.code_embeddings, candidates)

    # [b, c, k]
    scores = self.scoring_fn(
      tf.expand_dims(mention_embeddings, axis=2),  # [b, c, 1, dim],
      candidate_embeddings                         # [b, c, k, dim
    ) * candidate_mask

    # [b, c, k-1]
    gold_negative_scores = self.scoring_fn(
      tf.expand_dims(candidate_embeddings[:, :, 0], axis=-2), #[b, c, 1, dim]
      candidate_embeddings[:, :, 1:]                          #[b, c, k-1, dim]
    ) * candidate_mask[:, :, 1:]
    return scores, gold_negative_scores

  def scoring_fn(self, mention_embeddings, candidate_embeddings):
    emb_diff = mention_embeddings - candidate_embeddings
    score = tf.reduce_sum(
      emb_diff * emb_diff,
      axis=-1,
      keepdims=False,
      name='energy'
    )
    return -1.0 * score

  def _crnn(self, concept_embeddings, concept_lengths):
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

  def _concept_index(self, contextualized_embeddings, concept_head_idxs, concept_lengths):
    max_num_tokens = tf.shape(contextualized_embeddings)[1]
    max_concept_length = tf.cast(tf.reduce_max(concept_lengths), tf.int32)
    # [1, 1, max_concept_length]
    concept_range = tf.expand_dims(tf.expand_dims(tf.range(max_concept_length, dtype=tf.int32), axis=0), axis=0)
    # [bsize, max_num_concepts, 1]
    # [bsize, max_num_concepts, max_concept_length]
    concept_idxs = tf.expand_dims(tf.cast(concept_head_idxs, tf.int32), axis=-1) + concept_range

    # ensure valid indices for indices outside range (will be ignored by rnn due to lengths)
    # this can occur because max_concept_length + max concept start idx can overflow past
    # end of sequence, so just max at end and rnn will ignore using length info
    concept_idxs = tf.maximum(tf.minimum(concept_idxs, max_num_tokens - 1), 0)

    # [bsize, max_num_concepts, max_concept_length, emb_size]
    concept_embeddings = tf.gather(
      # [bsize, num_tokens, emb_size]
      contextualized_embeddings,
      # [bsize, max_num_concepts, max_concept_length]
      concept_idxs,
      batch_dims=1
    )
    return concept_embeddings


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
    init_string = ""
    if trainable_var.name in initialized_variable_names:
      init_string = '*INIT_FROM_CKPT*'
    print(f'{trainable_var.name}: {trainable_var.get_shape()} {init_string}')