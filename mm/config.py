import os
from hedgedog.tf.estimator.ingredients import *

model_ing = Ingredient('model', ingredients=[sampling_ingredient, dataset_ingredient])


# noinspection PyUnusedLocal
@dataset_ingredient.config
def ds_config():
  batch_size = 8
  project_dir = '/home/rmm120030/working/kge_ner'
  bert_model = 'ncbi_uncased_base'
  candidates_per_concept = 100
  dataset = 'medmentions'
  tagset = 'IOB'
  tasks = ['boundary', 'norm', 'type']
  ignore_sentences_without_concepts = False
  record_dir_name = 'iob.tf-records'


# noinspection PyUnusedLocal
@estimator_ingredient.config
def es_config(dataset):
  model_dir = os.path.join(dataset['project_dir'], 'model', dataset['dataset'])
  boundary_run = None
  boundary_ckpt = None
  norm_run = None
  norm_ckpt = None
  vram_allocation_strategy = 'dynamic'

  early_stopping = 'no-increase'
  metric_name = 'aggregate_metric'
  max_epochs_without_improvement = 5.0
  keep_checkpoint_max = 7


# noinspection PyUnusedLocal
@model_ing.config
def model_config(dataset):
  bert_model = dataset['bert_model']
  modules = dataset['tasks']
  verbose_eval = False
  umls_embeddings = 'max'

  # boundary
  boundary_weight = 1.

  # norm
  norm_weight = 1.
  span_pooling_strategy = 'mean'
  embedding_size = 50
  activation = 'gelu'
  scoring_fn = 'cos'
  norm_loss_fn = 'multinomial_ce'
  margin = 1.0
  learn_concept_embeddings = False
  use_string_sim = True
  informed_score_weighting = False

  # type
  type_weight = 0.1
  type_model = 'simple'
  type_layers = []
  type_activation = 'sigmoid'
  type_metric = None
  eval_thresholds = [0.6, 0.7, 0.8, 0.9]
  separate_type_embedding = False


# noinspection PyUnusedLocal
@training_ingredient.config
def train_config():
  learning_rate = 5e-5


############
# hooks
@dataset_ingredient.config_hook
def ds_config_hook(config, command_name, logger):
  ds = config['dataset']
  ds['ignore_sentences_without_concepts'] = 'boundary' not in ds['tasks']
  ds['data_dir'] = os.path.join(ds['project_dir'], f"{ds['dataset']}_dataset")
  return dataset_config_hook(config, command_name, logger)


@model_ing.config_hook
def model_save_config_hook(config, command_name, logger):
  return save_config_hook(config, command_name, logger, 'model')
