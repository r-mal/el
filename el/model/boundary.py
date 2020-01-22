import tensorflow as tf
from hedgedog.tf.estimator.multitask import Module
from hedgedog.tf.typing import TensorDict, TensorOrTensorDict
from hedgedog.nlp.seq_tag import get_tags
from tensorflow.contrib import crf
from hedgedog.tf import layers as hdlayers
from hedgedog.tf import metrics as hdmetrics
from hedgedog.logging import get_logger

from el.config import model_ing

log = get_logger("el.model.boundary")


class BoundaryModule(Module):
  @model_ing.capture
  def __init__(self, params, is_training, verbose_eval, use_bilstm, thiccness):
    super().__init__(params, is_training)
    tags = get_tags(params.dataset.tagset)
    self.boundary2id = tags.tag2id()
    d = len(self.boundary2id)
    self.crf_params = tf.get_variable("crf_params", shape=[d, d], dtype=tf.float32)
    self.verbose_eval = verbose_eval
    self.use_bilstm = use_bilstm
    self.thiccness = thiccness
    log.info(f"Initialized Boundary Module with tagset: {params.dataset.tagset} {self.boundary2id}")

  def __call__(self, shared_representation: TensorOrTensorDict, features: TensorDict) -> TensorDict:
    contextualized_tokens = shared_representation['contextualized_tokens']

    if self.thiccness > 0:
      last_four = shared_representation['all_layers'][-self.thiccness:]
      contextualized_tokens = tf.concat(last_four, axis=-1)

    if self.use_bilstm:
      cell_fw = tf.nn.rnn_cell.LSTMCell(hdlayers.get_shape_list(contextualized_tokens)[-1])
      cell_bw = tf.nn.rnn_cell.LSTMCell(hdlayers.get_shape_list(contextualized_tokens)[-1])
      # ([b, slen, dim*2], [b, slen, dim*2]) tuple
      # noinspection PyUnresolvedReferences
      outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, contextualized_tokens,
                                                             sequence_length=shared_representation['slens'],
                                                             dtype=tf.float32,
                                                             swap_memory=True)
      contextualized_tokens = tf.concat(outputs, axis=-1)

    logits = hdlayers.dense_with_layer_norm(contextualized_tokens,
                                            len(self.boundary2id))
    features['boundary_logits'] = logits
    features['slens'] = shared_representation['slens']
    return features

  def predict(self, graph_outputs_dict: TensorDict) -> TensorDict:
    graph_outputs_dict['predicted_boundaries'], _ =\
      crf.crf_decode(graph_outputs_dict['boundary_logits'],
                     self.crf_params,
                     graph_outputs_dict['slens'])
    return graph_outputs_dict

  def loss(self, graph_outputs_dict: TensorDict, labels: TensorDict) -> TensorOrTensorDict:
    log_likelihood, _ = crf.crf_log_likelihood(inputs=graph_outputs_dict['boundary_logits'],
                                               tag_indices=labels['boundaries'],
                                               sequence_lengths=graph_outputs_dict['slens'],
                                               transition_params=self.crf_params)
    loss = -tf.reduce_mean(log_likelihood)
    return {'boundary_loss': loss}

  def eval_metrics(self, graph_outputs_dict: TensorDict, labels: TensorDict, loss: TensorOrTensorDict) -> TensorDict:
    predictions = self.predict(graph_outputs_dict)['predicted_boundaries']
    labels = labels['boundaries']
    k = len(self.boundary2id)
    weights = tf.sequence_mask(graph_outputs_dict['slens'], dtype=tf.float32)
    metric_ops = {'boundary/f1': hdmetrics.avg_f1(labels, predictions, k,
                                                  pos_indices=list(range(1, k)),
                                                  weights=weights),
                  'boundary/accuracy': tf.metrics.accuracy(labels, predictions, weights)}
    if self.verbose_eval:
      metric_ops = {
        **metric_ops,
        **hdmetrics.multiclass_f1(labels, predictions, self.boundary2id, mask=weights, group='boundary'),
        'boundary/prec': hdmetrics.avg_precision(labels, predictions, k,
                                                 pos_indices=list(range(1, k)),
                                                 weights=weights),
        'boundary/recall': hdmetrics.avg_recall(labels, predictions, k,
                                                pos_indices=list(range(1, k)),
                                                weights=weights)
      }

    return metric_ops
