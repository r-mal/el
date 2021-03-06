import json
from pathlib import Path
import os
import tensorflow as tf
import numpy as np
from hedgedog.tf.typing import TensorDict, TensorOrTensorDict
from hedgedog.tf import layers as hdlayers
from hedgedog.tf import metrics as hdmetrics

from el.config import model_ing
from el.model.normalization import RankingModule


class TypingModule(RankingModule):
  @model_ing.capture
  def __init__(self, params, is_training, type_loss_fn: str):
    super().__init__(params, is_training)
    self.tui2label_id = json.load((Path(params.dataset.project_dir) / 'info' / 'tui2label.json').open())
    self.weight = params.model.type_weight
    self.activation = params.model.type_activation
    self.metric = params.model.type_metric
    self.loss_fn = {
      'multinomial_ce': self.multinomial_cross_entropy,
      'multinomial_ce_prob': self.multinomial_cross_entropy_prob,
      'pointwise_margin_loss': self.pointwise_margin_loss,
      'margin_loss': self.margin_loss,
      'energy_loss': self.energy_loss
    }[type_loss_fn]

  def __call__(self, shared_representation: TensorOrTensorDict, features: TensorDict) -> TensorDict:
    # [b, c, l]
    features['type_probs'] = tf.layers.dense(features['mention_embeddings'], len(self.tui2label_id),
                                             activation=self.activation)

    return features

  def predict(self, graph_outputs_dict: TensorDict) -> TensorDict:
    return graph_outputs_dict

  def loss(self, graph_outputs_dict: TensorDict, labels: TensorDict) -> TensorOrTensorDict:
    # [b, c, l]
    labels = tf.cast(labels['semtype_labels'], tf.float32)
    anti_labels = tf.cast(tf.not_equal(labels, 1), tf.float32)
    positive_scores = graph_outputs_dict['type_probs'] * labels
    negative_scores = graph_outputs_dict['type_probs'] * anti_labels

    # [b, c]
    losses = self.loss_fn(positive_scores, negative_scores, labels, anti_labels)
    # [b, c]
    concept_mask = tf.sequence_mask(graph_outputs_dict['num_concepts'], dtype=tf.float32)

    loss = tf.reduce_sum(losses * concept_mask) / tf.maximum(tf.reduce_sum(concept_mask), 1)
    for s_loss in self.secondary_losses:
      loss += tf.reduce_sum(s_loss * concept_mask) / tf.maximum(tf.reduce_sum(concept_mask), 1)

    return {"typing_loss": loss * self.weight}

  def eval_metrics(self, graph_outputs_dict: TensorDict, labels: TensorDict, loss: TensorOrTensorDict) -> TensorDict:
    # [b, c, l]
    type_labels = tf.cast(labels['semtype_labels'], tf.bool)
    # [b, c, l]
    type_probs = graph_outputs_dict['type_probs']
    # [b, c]
    concept_mask = tf.expand_dims(tf.sequence_mask(graph_outputs_dict['num_concepts'], dtype=tf.bool), axis=-1)

    eval_metric_ops = {}
    for cutoff in self.params.model.eval_thresholds:
      predictions = tf.greater_equal(type_probs, cutoff)
      weights = tf.logical_or(type_labels, predictions)
      eval_metric_ops[f'type/f1@{cutoff}'] = hdmetrics.f1_score(labels=type_labels,
                                                                predictions=predictions,
                                                                weights=weights)
      eval_metric_ops[f'type/hot@{cutoff}'] = hdmetrics.count(predictions, weights=concept_mask)

      correct = tf.reduce_prod(tf.cast(tf.equal(predictions, type_labels), dtype=tf.int32), axis=-1)
      eval_metric_ops[f'type/acc@{cutoff}'] = tf.metrics.accuracy(tf.ones_like(correct), correct)
    eval_metric_ops['type/num_hot'] = hdmetrics.count(type_labels, weights=concept_mask)
    eval_metric_ops['type/loss'] = tf.metrics.mean(loss['typing_loss'] / self.weight)

    type_labels = tf.cast(labels['semtype_labels'], tf.float32)
    neg_labels = tf.ones_like(type_labels) - type_labels
    eval_metric_ops['type/avg_pos_prob'] = tf.metrics.mean(type_probs, weights=type_labels)
    eval_metric_ops['type/avg_neg_prob'] = tf.metrics.mean(type_probs, weights=neg_labels)
    return eval_metric_ops

  # noinspection PyMethodOverriding
  def pointwise_margin_loss(self, pos_scores, negative_scores, pmask, nmask):
    # [b, k, n]
    rectified_scores = tf.nn.relu(negative_scores) * nmask
    # [b, k, p]
    rectified_scores += tf.nn.relu(self.margin - pos_scores) * pmask

    return tf.reduce_sum(rectified_scores, axis=-1)

  # noinspection PyMethodOverriding
  def margin_loss(self, pos_scores, negative_scores, pmask, nmask):
    # [b, k, p, 1]
    pos_scores = tf.expand_dims(pos_scores, axis=-1)
    pmask = tf.expand_dims(pmask, axis=-1)
    # [b, k, 1, n]
    negative_scores = tf.expand_dims(negative_scores, axis=-2)
    nmask = tf.expand_dims(nmask, axis=-2)
    # [b, k, p, n]
    losses = tf.nn.relu(self.margin - pos_scores + negative_scores) * pmask * nmask
    # [b, k]
    # TODO probably should sum then divide here
    return tf.reduce_mean(tf.reduce_sum(losses, axis=-1), axis=-1)



  # noinspection PyMethodOverriding
  def multinomial_cross_entropy(self, positive_scores, negative_scores, pmask, nmask):
    # [b, k, n + p]
    exp_scores = tf.exp(tf.concat((negative_scores, positive_scores), axis=-1))
    # [b, k, n + p]
    mask = tf.concat((nmask, pmask), axis=-1)
    exp_scores *= mask
    # [b, k]
    neg_logsumexp = tf.log(tf.reduce_sum(exp_scores, axis=-1))

    return neg_logsumexp - tf.reduce_sum(positive_scores, axis=-1)


class TypeEmbeddingModule(TypingModule):
  @model_ing.capture
  def __init__(self, params, is_training, type_text_embeddings):
    super().__init__(params, is_training)
    self.separate_type_embedding = params.model.separate_type_embedding
    self.embedding_size = params.model.embedding_size

    # init type embeddings
    with np.load(os.path.join(params.dataset.project_dir, 'info',
                              params.model.umls_embeddings, f'{type_text_embeddings}.npz')) as npz:
      # [l, dim]
      self.type_embeddings = tf.Variable(npz['embs'], trainable=False, name='type_embeddings', dtype=tf.float32)
      self.embedding_dim = self.type_embeddings.shape[-1]
    self.layers = params.model.type_layers
    if len(self.layers) > 0 and self.layers[-1] != self.embedding_dim:
      self.layers += [self.embedding_dim]

  def __call__(self, shared_representation: TensorOrTensorDict, features: TensorDict) -> TensorDict:
    # [b, c, dim]
    # if self.separate_type_embedding:
    #   mention_embeddings = hdlayers.dense_with_layer_norm(features['pooled_mention'],
    #                                                       self.embedding_size,
    #                                                       self.activation,
    #                                                       drop_prob=self.drop_prob)
    # else:
    #   mention_embeddings = features['mention_embeddings']
    # for dim in self.layers:
    #   mention_embeddings = hdlayers.dense_with_layer_norm(mention_embeddings, dim,
    #                                                       activation=self.activation,
    #                                                       drop_prob=self.drop_prob)
    # # [l, dim]
    # # type_embeddings
    # # [b, c, dim]
    # mention_embeddings = tf.nn.l2_normalize(mention_embeddings, axis=-1)
    concept_encodings = features['pooled_mention']
    concept_type_embeddings = tf.layers.dense(
      inputs=concept_encodings,
      units=self.embedding_size,
      activation=None,
      name='embeddings_type'
    )
    embeddings_norm = tf.norm(concept_type_embeddings, ord=2, axis=-1, keepdims=True)
    concept_type_embeddings = concept_type_embeddings / tf.maximum(embeddings_norm, 1.0)
    # [b, c, 1, d]
    concept_type_embeddings = tf.expand_dims(concept_type_embeddings, axis=-2)
    # [1, 1, k, d]
    type_embeddings = tf.expand_dims(tf.expand_dims(self.type_embeddings, axis=-0), axis=0)
    features['type_probs'] = self.scoring_fn(concept_type_embeddings, type_embeddings)

    # [b, c, l]
    # features['type_probs'] = tf.matmul(mention_embeddings, self.type_embeddings, transpose_b=True)

    return features
