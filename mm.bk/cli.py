from hedgedog import logging
from sacred import Experiment
from hedgedog.tf.estimator.ingredients import *
from hedgedog.tf.io.dataset_util import inspect_tfrecords, iterate_records

from mm.data.dataset import MMDataset
# noinspection PyUnresolvedReferences
from mm import config as _config

logging.reset_handlers()
log = logging.get_logger('mm')

ex = Experiment(ingredients=[dataset_ingredient])


##########################
# Data
##########################
@ex.command



@ex.command
def preprocess():
  ds = MMDataset()
  ds.preprocess()


@ex.command
def inspect():
  ds = MMDataset()
  inspect_tfrecords(ds.load_train(batch_size=2), max_size=10, visualize_fn=ds.create_print_fn())
  log.info(f"Iterated {iterate_records(ds.load_train(repeat=False)) * ds.batch_size} train records")
  inspect_tfrecords(ds.load_dev(batch_size=2), max_size=10, visualize_fn=ds.create_print_fn())
  log.info(f"Iterated {iterate_records(ds.load_dev()) * ds.batch_size} dev records")
  inspect_tfrecords(ds.load_test(batch_size=2), max_size=10, visualize_fn=ds.create_print_fn())
  log.info(f"Iterated {iterate_records(ds.load_test()) * ds.batch_size} dev records")


@ex.automain
def main():
  pass
