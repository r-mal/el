import tensorflow as tf
import numpy as np
from hedgedog.tf.estimator.multitask import Module
from hedgedog.tf.typing import TensorDict, TensorOrTensorDict
from hedgedog.tf import layers as hdlayers
from hedgedog.tf import metrics as hdmetrics
from hedgedog.logging import get_logger

from mm.config import model_ing

log = get_logger("mm.model.norm")


class NormalizationModule(Module):
  @model_ing.capture
  def __init__(self, params, is_training,
               span_pooling_strategy: str, embedding_size: int, activation: str, scoring_fn: str,
               norm_loss_fn: str, margin: float, learn_concept_embeddings: bool, umls_embeddings: str,
               use_string_sim: bool, informed_score_weighting: bool):
    super().__init__(params, is_training)
    self.span_pooling_strategy = span_pooling_strategy
    self.embedding_size = embedding_size
    self.activation = activation
    self.margin = margin
    self.use_string_sim = use_string_sim
    self.scoring_fn = {
      'cos': hdlayers.cos_sim,
      'dot': lambda x, y: tf.reduce_sum(x * y, axis=-1)
    }[scoring_fn]
    self.loss_fn = {
      'multinomial_ce': multinomial_cross_entropy,
      'pointwise_margin_loss': self.pointwise_margin_loss,
      'margin_loss': self.margin_loss
    }[norm_loss_fn]

    self.code_embeddings = self._init_embeddings(params, umls_embeddings, learn_concept_embeddings)

    # score weights
    self.informed_score_weighting = informed_score_weighting
    if not self.informed_score_weighting:
      with tf.variable_scope('score', reuse=tf.AUTO_REUSE):
        # self.embedding_weight = tf.get_variable('emb_match_weight', shape=[], dtype=tf.float32)
        # self.string_weight = tf.get_variable('str_match_weight', shape=[], dtype=tf.float32)
        self.embedding_weight = tf.Variable(0.5, name='emb_match_weight')
        self.string_weight = tf.Variable(0.5, name='str_match_weight')

    log.info(f"Initialized Normalization module with {'joint' if use_string_sim else 'embedding'} similarity"
             f" and {'informed' if informed_score_weighting else 'uninformed'} score weighting")

  def _init_embeddings(self, params, umls_embeddings, learn_concept_embeddings):
    import json, os
    with np.load(os.path.join(params.dataset.project_dir, 'info', umls_embeddings, 'embeddings.npz')) as npz:
      np_code_embeddings = npz['embs']
    if learn_concept_embeddings:
      code_embeddings = tf.get_variable('umls_embeddings', shape=np_code_embeddings.shape, dtype=tf.float32,
                                             trainable=True)
    else:
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
    tokens = shared_representation['contextualized_tokens']
    _, n, d = hdlayers.get_shape_list(tokens)
    # [b, c, n]
    concept_token_masks = features['concept_token_masks']
    _, c, _ = hdlayers.get_shape_list(concept_token_masks)

    # pool tokens of each concept mention
    # [b, 1, n, d]
    tokens = tf.expand_dims(tokens, axis=1)
    # [b, c, n, 1]
    concept_token_masks = tf.expand_dims(concept_token_masks, axis=-1)

    # [b, c, n, d]
    concept_token_embeddings = tf.tile(tokens, [1, c, 1, 1]) * tf.tile(concept_token_masks, [1, 1, 1, d])

    # reduce mean
    # [b, c, d]
    mean_pooled = hdlayers.masked_mean_pooling(concept_token_embeddings, concept_token_masks,
                                               reduction_index=2,
                                               expand_mask=False)

    # project into embedding space
    mention_embeddings = hdlayers.dense_with_layer_norm(mean_pooled, self.embedding_size, self.activation,
                                                        drop_prob=self.drop_prob)

    if self.informed_score_weighting:
      # calculate embedding relevance scores
      max_pooled = tf.reduce_max(concept_token_embeddings, axis=2)
      features['embedding_weight'] = tf.layers.dense(max_pooled, 1, activation=tf.nn.sigmoid)
    else:
      features['embedding_weight'] = tf.nn.sigmoid(self.embedding_weight)

    features['pooled_mention'] = mean_pooled
    features['mention_embeddings'] = mention_embeddings
    return features

  def predict(self, graph_outputs_dict: TensorDict) -> TensorDict:
    if 'candidates' in graph_outputs_dict:
      graph_outputs_dict['candidate_scores'] = self._calc_scores(graph_outputs_dict, graph_outputs_dict)

    graph_outputs_dict.__delitem__('embedding_weight')
    return graph_outputs_dict

  def loss(self, graph_outputs_dict: TensorDict, labels: TensorDict) -> TensorOrTensorDict:
    # pos_score, negative_scores = self._calc_scores(graph_outputs_dict, labels)
    scores = self._calc_scores(graph_outputs_dict, labels)
    pos_score = scores[:, :, 0]
    negative_scores = scores[:, :, 1:]

    # [b, c]
    losses = self.loss_fn(pos_score, negative_scores)
    concept_mask = tf.sequence_mask(graph_outputs_dict['num_concepts'], dtype=tf.float32)
    loss = tf.reduce_mean(losses * concept_mask)

    return {"normalization_loss": loss * self.params.model.norm_weight}

  def eval_metrics(self, graph_outputs_dict: TensorDict, labels: TensorDict, loss: TensorOrTensorDict) -> TensorDict:
    # pos_score, negative_scores = self._calc_scores(graph_outputs_dict, labels)
    # [b, c, k+1]
    scores = self._calc_scores(graph_outputs_dict, labels)

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
      # 'normalization/mean_emb_weight': tf.metrics.mean(graph_outputs_dict['embedding_weight']),
      'normalization/str_weight': tf.metrics.mean(self.string_weight),
      'normalization/emb_weight': tf.metrics.mean(self.embedding_weight)
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
    # [b, c, k, dim]
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

    if self.use_string_sim:
      # [b, c, k]
      candidate_scores = labels['candidate_scores']
      # embedding_weight = graph_outputs_dict['embedding_weight']
      # string_weight = 1. - embedding_weight
      # scores = (string_weight * candidate_scores) + (embedding_weight * scores)
      scores = (self.string_weight * candidate_scores) + (self.embedding_weight * scores)

    return scores

  def pointwise_margin_loss(self, pos_score, negative_scores):
    # [b, k, c]
    rectified_scores = tf.nn.relu(negative_scores)
    # [b, k]
    pos = tf.nn.relu(self.margin - pos_score)

    return pos + tf.reduce_sum(rectified_scores, axis=-1)

  def margin_loss(self, pos_score, negative_scores):
    # [b, k, 1]
    pos_scores = tf.expand_dims(pos_score, axis=-1)
    # [b, k, c]
    losses = tf.nn.relu(self.margin - pos_scores + negative_scores)
    # [b, k]
    return tf.reduce_sum(losses, axis=-1)


def multinomial_cross_entropy(pos_score, negative_scores):
  """
  Performs multinomial cross-entropy
  - sum_p[p] + log sum_n[exp(n)]
  :param pos_score: [b, k] tensor of scores for positive example scores
  :param negative_scores: [b, k, n] tensor of scores for negative examples
  :return: [b, k] loss tensor
  """
  # [b, k, c+1]
  # candidate_scores = tf.concat((negative_scores, tf.expand_dims(pos_score, axis=-1)), axis=-1)
  # [b, k] this is essentially the maximum candidate score
  neg_logsumexp = tf.log(tf.reduce_sum(tf.exp(negative_scores), axis=-1))

  return neg_logsumexp - pos_score
