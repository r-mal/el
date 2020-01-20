from hedgedog.tf.estimator.ingredients import *


# noinspection PyUnusedLocal
@dataset_ingredient.config
def ds_config():
  batch_size = 8
  project_dir = '/home/rmm120030/working/mm'
  data_dir = '/home/rmm120030/working/mm/dataset'
  bert_model = 'base_uncased'
  candidates_per_concept = 9
