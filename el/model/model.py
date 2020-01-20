import typing
from argparse import Namespace
import tensorflow as tf
import tensorflow_estimator as tfe
from hedgedog.tf.io.dataset import Dataset
from hedgedog.tf.models.multitask_bert_model import MultitaskBertModel
from hedgedog.tf.typing import TensorOrTensorDict
from hedgedog.logging import get_logger

from el.data.dataset import NerDataset
from el.config import model_ing
from el.model.boundary import BoundaryModule
from el.model.normalization import NormalizationModule
from el.model.type import TypingModule, TypeEmbeddingModule

log = get_logger("el.model")


class MTBertModel(MultitaskBertModel):
  @model_ing.capture
  def __init__(self, mode: str, hyperparameters: Namespace, bert_model: str, dataset: Dataset = None):
    is_training = mode == "TRAIN"
    prediction_modules = []
    self.module_names = hyperparameters.model.modules
    if 'boundary' in hyperparameters.model.modules:
      prediction_modules.append(BoundaryModule(hyperparameters, is_training))
    if 'norm' in hyperparameters.model.modules:
      prediction_modules.append(NormalizationModule(hyperparameters, is_training))
    if 'type' in hyperparameters.model.modules:
      type_model_cons = {
        'simple': TypingModule,
        'embedding': TypeEmbeddingModule
      }[hyperparameters.model.type_model]
      prediction_modules.append(type_model_cons(hyperparameters, is_training))
    assert len(prediction_modules) > 0, print(f"No valid modules in: {hyperparameters.model.modules}")
    log.info(f"Initialized MTBertModel with modules {self.module_names}")
    super().__init__(mode, hyperparameters, modules=prediction_modules, bert_model=bert_model, dataset=dataset)

  def compute_aggregate_metric_ops(self, metric_ops):
    metrics = []
    ops = []
    if 'boundary' in self.module_names and self.params.model.boundary_weight > 0:
      metrics.append(metric_ops['boundary/f1'][0])
      ops.append(metric_ops['boundary/f1'][1])
    if 'norm' in self.module_names and self.params.model.norm_weight > 0:
      metrics.append(metric_ops['normalization/strict_accuracy'][0])
      ops.append(metric_ops['normalization/strict_accuracy'][1])
    if 'type' in self.module_names and self.params.model.type_weight > 0 and self.params.model.type_metric is not None:
      metrics.append(metric_ops[f'type/{self.params.model.type_metric}'][0])
      ops.append(metric_ops[f'type/{self.params.model.type_metric}'][1])

    if len(metrics) > 0:
      aggregate_metric = sum(metrics) / len(metrics)
    else:
      return {}

    return {'aggregate_metric': (aggregate_metric, tf.group(ops))}

  @staticmethod
  def dataset() -> Dataset:
    return NerDataset()

  def export_outputs(self, graph_outputs: TensorOrTensorDict, predictions: TensorOrTensorDict)\
      -> typing.Dict[str, tfe.estimator.export.ExportOutput]:
    pass
