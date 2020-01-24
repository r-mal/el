from abc import ABC

import tensorflow as tf
import numpy as np
import os
from hedgedog.tf.estimator.multitask import Module
from hedgedog.tf.typing import TensorDict, TensorOrTensorDict
from hedgedog.tf import layers as hdlayers
from hedgedog.tf import metrics as hdmetrics
from hedgedog.logging import get_logger
import json

from el.config import model_ing

log = get_logger("el.model.norm")


class RankingModule(Module, ABC):
  @model_ing.capture
  def __init__(self, params, is_training, scoring_fn: str, norm_loss_fn: str, margin: float, offline_emb_strat: str,
               offline_emb_file: str = None):
    super().__init__(params, is_training)
    self.margin = margin
    self.scoring_fn = {
      'cos': hdlayers.cos_sim,
      'dot': lambda x, y: tf.reduce_sum(x * y, axis=-1),
      'energy': self.energy,
      'energy_with_loss': self.energy_with_loss
    }[scoring_fn]
    self.loss_fn = {
      'multinomial_ce': self.multinomial_cross_entropy,
      'multinomial_ce_prob': self.multinomial_cross_entropy_prob,
      'pointwise_margin_loss': self.pointwise_margin_loss,
      'margin_loss': self.margin_loss,
      'energy_loss': self.energy_loss
    }[norm_loss_fn]

    # offline mention embeddings
    self.offline_emb_strat = offline_emb_strat
    if offline_emb_strat is not None:
      if offline_emb_file is None:
        offline_emb_file = os.path.join(params.dataset.project_dir, 'info', f'{params.dataset.dataset}_mentions.npz')
      with np.load(offline_emb_file) as npz:
        offline_mention_embeddings = npz['embs']
      self.offline_mention_embeddings = tf.Variable(offline_mention_embeddings,
                                                    trainable=False,
                                                    name='offline_mention_embeddings')
      self.offline_emb_weight = tf.Variable(0.5, name='offline_match_weight')
    self.secondary_losses = {}

  def pointwise_margin_loss(self, pos_score, negative_scores, labels):
    # [b, k, c]
    rectified_scores = tf.nn.relu(negative_scores)
    # [b, k]
    pos = tf.nn.relu(self.margin - pos_score)

    return pos + tf.reduce_sum(rectified_scores, axis=-1)

  def margin_loss(self, pos_score, negative_scores, labels):
    # [b, k, 1]
    pos_scores = tf.expand_dims(pos_score, axis=-1)
    # [b, k, c]
    losses = tf.nn.relu(self.margin - pos_scores + negative_scores)

    loss_candidate_count = tf.maximum(tf.reduce_sum(labels['candidate_mask'][:, :, 1:], axis=-1), 1)
    # [b, k]
    losses = tf.reduce_sum(losses * labels['candidate_mask'][:, :, 1:], axis=-1) / loss_candidate_count
    # [b, k]
    return losses


  # noinspection PyMethodMayBeStatic
  def multinomial_cross_entropy(self, pos_score, negative_scores, labels):
    """
    Performs multinomial cross-entropy
    - sum_p[p] + log sum_n[exp(n)]
    :param pos_score: [b, k] tensor of scores for positive example scores
    :param negative_scores: [b, k, n] tensor of scores for negative examples
    :return: [b, k] loss tensor
    """
    # [b, k, c+1]
    candidate_scores = tf.concat((negative_scores, tf.expand_dims(pos_score, axis=-1)), axis=-1)
    # [b, k]
    neg_logsumexp = tf.log(tf.reduce_sum(tf.exp(candidate_scores), axis=-1))
    # [b, k]
    loss = neg_logsumexp - pos_score
    return loss

  # noinspection PyMethodMayBeStatic
  def multinomial_cross_entropy_prob(self, pos_probs, negative_probs, labels):
    # [b, k]

    return -tf.log(pos_probs + 1e-12)

  def energy_loss(self, pos_score, negative_scores, labels):
    loss = -pos_score
    return loss

  def energy(self, x, y):
    emb_diff = x - y
    distance = tf.reduce_sum(
      emb_diff * emb_diff,
      axis=-1,
      keepdims=False,
      name='energy'
    )
    # d = 2 - 2cos(x,y)
    # (d - 2)/(-2) = cos(x, y)
    # 2-(d/2) = cos(x, y)
    score = (2 - (0.5 * distance))
    return score

  def energy_with_loss(self, x, y):
    emb_diff = x - y
    distance = tf.reduce_sum(
      emb_diff * emb_diff,
      axis=-1,
      keepdims=False,
      name='energy'
    )
    # only true energy, others are wrong
    self.secondary_losses['energy_loss'] = distance[:, :, 0]
    # d = 2 - 2cos(x,y)
    # (d - 2)/(-2) = cos(x, y)
    # 2-(d/2) = cos(x, y)
    score = (2 - (0.5 * distance))
    return score


class NormalizationModule(RankingModule):
  @model_ing.capture
  def __init__(self, params, is_training,
               span_pooling_strategy: str, embedding_size: int, activation: str, learn_concept_embeddings: bool,
               umls_embeddings: str, use_string_sim: bool, informed_score_weighting: bool, string_method: str):
    super().__init__(params, is_training)
    self.span_pooling_strategy = span_pooling_strategy
    self.embedding_size = embedding_size
    self.activation = activation
    self.use_string_sim = use_string_sim
    self.string_method = string_method
    self.code_embeddings = self._init_embeddings(params, umls_embeddings, learn_concept_embeddings)

    # score weights
    self.informed_score_weighting = informed_score_weighting
    if not self.informed_score_weighting:
      with tf.variable_scope('score', reuse=tf.AUTO_REUSE):
        self.embedding_weight = tf.Variable(0.5, name='emb_match_weight')
        self.string_weight = tf.Variable(0.5, name='str_match_weight')
        self.embedding_bias = tf.Variable(0.0, name='score_bias')
        self.string_bias = tf.Variable(0.0, name='score_bias')

    log.info(f"Initialized Normalization module with {'joint' if use_string_sim else 'embedding'} similarity"
             f" and {'informed' if informed_score_weighting else 'uninformed'} score weighting")

  # noinspection PyMethodMayBeStatic
  def _init_embeddings(self, params, umls_embeddings, learn_concept_embeddings):
    with np.load(os.path.join(params.dataset.project_dir, 'info', umls_embeddings, 'embeddings.npz')) as npz:
      np_code_embeddings = npz['embs']
    if learn_concept_embeddings:
      code_embeddings = tf.get_variable('umls_embeddings', shape=np_code_embeddings.shape, dtype=tf.float32,
                                        trainable=True)
    else:
      code_embeddings = tf.Variable(np_code_embeddings, trainable=False, name='umls_embeddings')

    code2id = json.load(open(os.path.join(params.dataset.project_dir, 'info', 'cui2id.json')))
    num_learnable_codes = len(code2id) - np_code_embeddings.shape[0]

    log.warning(f'{num_learnable_codes} Learnable concepts added to embeddings. '
                f'{len(code2id)} vs {np_code_embeddings.shape[0]}')

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
    if self.offline_emb_strat == 'concat':
      offline_embeddings = tf.nn.embedding_lookup(self.offline_mention_embeddings, features['mention_embedding_idx'])
      mean_pooled = tf.concat((mean_pooled, offline_embeddings), axis=-1)

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
    # TODO split these losses up so its easier to monitor
    losses = self.loss_fn(pos_score, negative_scores, labels)
    concept_mask = tf.sequence_mask(graph_outputs_dict['num_concepts'], dtype=tf.float32)
    # TODO figure out if I should avg across sentences or just do this where every concept contributes equally
    losses = losses * concept_mask
    # mean over actual concepts
    loss = tf.reduce_sum(losses, axis=-1) / tf.maximum(tf.reduce_sum(concept_mask, axis=-1), 1)
    # mean over batch
    loss = tf.reduce_mean(loss)
    loss_collection = {
      "normalization_loss": loss * self.params.model.norm_weight
    }

    for s_loss_name, s_loss_value in self.secondary_losses.items():
      s_loss_value = s_loss_value * self.params.model.norm_weight
      s_loss_value = s_loss_value * concept_mask
      s_loss_value = tf.reduce_sum(s_loss_value, axis=-1) / tf.maximum(tf.reduce_sum(concept_mask, axis=-1), 1)
      s_loss_value = tf.reduce_mean(s_loss_value)
      loss_collection[s_loss_name] = s_loss_value
    return loss_collection

  def eval_metrics(self, graph_outputs_dict: TensorDict, labels: TensorDict, loss: TensorOrTensorDict) -> TensorDict:
    # pos_score, negative_scores = self._calc_scores(graph_outputs_dict, labels)
    # [b, c, k+1]
    scores = self._calc_scores(graph_outputs_dict, labels)

    # [b, c]
    ones = tf.sequence_mask(graph_outputs_dict['num_concepts'], dtype=tf.bool)
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

    # [bsize, c]
    maximum_negative_score = tf.reduce_max(scores[:, :, 1:], axis=-1)

    eval_metrics = {
      self.params.model.norm_loss_fn: tf.metrics.mean(loss['normalization_loss']),
      'normalization/accuracy': accuracy,
      'normalization/normalized_accuracy': normalized_accuracy,
      'normalization/strict_accuracy': strict_accuracy,
      'normalization/avg_pos_score': tf.metrics.mean(scores[:, :, 0], weights=ones),
      'normalization/avg_neg_score': tf.metrics.mean(scores[:, :, 1:], weights=labels['candidate_mask'][:, :, 1:]),
      'normalization/avg_max_neg_score': tf.metrics.mean(maximum_negative_score, weights=ones),
      # 'normalization/mean_emb_weight': tf.metrics.mean(graph_outputs_dict['embedding_weight']),
      'normalization/str_weight': tf.metrics.mean(self.string_weight),
      'normalization/string_bias': tf.metrics.mean(self.string_bias),
      'normalization/emb_weight': tf.metrics.mean(self.embedding_weight),
      'normalization/embedding_bias': tf.metrics.mean(self.embedding_bias),
    }

    if self.offline_emb_strat is not None:
      eval_metrics['normalization/offline_emb_weight'] = self.offline_emb_weight

    for secondary_loss_name in self.secondary_losses.keys():
      eval_metrics[secondary_loss_name] = tf.metrics.mean(loss[secondary_loss_name])

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
    )

    if self.use_string_sim:
      if self.string_method == 'weighted_scores':
        # [b, c, k]
        candidate_scores = labels['candidate_scores']
        scores = (self.string_weight * candidate_scores) + (self.embedding_weight * scores) + self.string_bias
      elif self.string_method == 'bayesian':
        # prior scores
        # [b, c, k]
        prior_scores = labels['candidate_scores']
        prior_weight = self.string_weight
        prior_bias = self.string_bias
        # prior_prob = tf.nn.softmax((prior_weight * prior_scores) + prior_bias, axis=-1)
        prior_logits = (prior_weight * prior_scores) + prior_bias
        prior_energies = tf.exp(prior_logits) * candidate_mask
        # [b, c, k]
        prior_prob = prior_energies / (tf.reduce_sum(prior_energies, axis=-1, keepdims=True) + 1e-12)

        # posterior scores
        # [b, c, k]
        likelihood_scores = scores
        likelihood_weight = self.embedding_weight
        likelihood_bias = self.embedding_bias
        # likelihood_prob = tf.nn.softmax(, axis=-1)
        likelihood_logits = (likelihood_weight * likelihood_scores) + likelihood_bias
        likelihood_energies = tf.exp(likelihood_logits) * candidate_mask
        # [b, c, k]
        likelihood_prob = likelihood_energies / (tf.reduce_sum(likelihood_energies, axis=-1, keepdims=True) + 1e-12)

        # [b, c, k]
        posterior_prob = likelihood_prob * prior_prob
        # [b, c, 1]
        normalized_posterior_prob = tf.reduce_sum(posterior_prob, axis=-1, keepdims=True)
        posterior_prob = posterior_prob / (normalized_posterior_prob + 1e-12)
        scores = posterior_prob
      elif self.string_method == 'mixture':
        # [b, c, k]
        prior_scores = labels['candidate_scores']
        prior_weight = self.string_weight
        prior_bias = self.string_bias
        prior_energies = tf.exp((prior_weight * prior_scores) + prior_bias) * candidate_mask

        # posterior scores
        # [b, c, k]
        likelihood_scores = scores
        likelihood_weight = self.embedding_weight
        likelihood_bias = self.embedding_bias
        likelihood_energies = tf.exp((likelihood_weight * likelihood_scores) + likelihood_bias) * candidate_mask

        # [b, k]
        normalization_term = tf.reduce_sum(prior_energies, axis=-1, keepdims=True)
        normalization_term += tf.reduce_sum(likelihood_energies, axis=-1, keepdims=True)

        posterior_energy = likelihood_energies + prior_energies
        # [b, c, k]
        posterior_prob = posterior_energy / (normalization_term + 1e-12)
        # [b, c, 1]
        scores = posterior_prob

      else:
        raise ValueError(f'String method not found: {self.string_method}')

    if self.offline_emb_strat == 'score':
      # [b, c, dim]
      offline_embeddings = tf.nn.embedding_lookup(self.offline_mention_embeddings,
                                                  graph_outputs_dict['mention_embedding_idx'])
      offline_scores = self.scoring_fn(tf.expand_dims(offline_embeddings, axis=2),
                                       candidate_embeddings) * candidate_mask
      scores += self.offline_emb_weight * offline_scores

    scores = scores * candidate_mask
    return scores
