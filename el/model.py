import typing
from argparse import Namespace
from hedgedog.tf.estimator.multitask import Module
from hedgedog.tf.io.dataset import Dataset
from hedgedog.tf.models.multitask_bert_model import MultitaskBertModel
from hedgedog.tf.typing import TensorOrTensorDict

from mm.data.dataset import MMDataset


class MTBertModel(MultitaskBertModel):
  def __init__(self, mode: str, hyperparameters: Namespace, modules: typing.List[Module], bert_model: str):
    super().__init__(mode, hyperparameters, modules, bert_model)

  @staticmethod
  def dataset() -> Dataset:
    return MMDataset()

  def export_outputs(self, graph_outputs: TensorOrTensorDict, predictions: TensorOrTensorDict) -> typing.Dict[
    str, tfe.estimator.export.ExportOutput]:
    pass
