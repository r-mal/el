from collections import defaultdict

from hedgedog import logging
from sacred import Experiment
from hedgedog.tf.estimator.ingredients import *
from hedgedog.tf.io.dataset_util import inspect_tfrecords, iterate_records
from hedgedog.tf.estimator import train as hdtrain
from hedgedog.tf.sacred import convert_to_namespace

from el.data.dataset import NerDataset
# noinspection PyUnresolvedReferences
from el import config as conf
from el.model.model import MTBertModel
from el.data import clef, medmentions
from el.eval.boundary import boundary_eval
from el.eval.entity import end_to_end_eval

logging.reset_handlers()
log = logging.get_logger('el')

ex = Experiment(ingredients=[sampling_ingredient, dataset_ingredient, estimator_ingredient,
                             conf.model_ing, training_ingredient])


@ex.command
def train(_run):
  params = convert_to_namespace(_run.config)
  hdtrain.train(_run, model_class=MTBertModel, parameters=params)


@ex.command
def evaluate(_run):
  log.info(f"Running eval with config: {_run.config}")
  params = convert_to_namespace(_run.config)
  hdtrain.evaluate(model_class=MTBertModel, parameters=params)


@ex.command
def boundary_evaluation(_run):
  params = convert_to_namespace(_run.config)
  boundary_eval(MTBertModel, params)


@ex.command
def end_to_end_evaluation(_run):
  params = convert_to_namespace(_run.config)
  end_to_end_eval(MTBertModel, params)


##########################
# Data
##########################
@ex.command
def create_clef_dataset():
  clef.create_dataset()


@ex.command
def create_medmentions_dataset():
  medmentions.create_dataset()


@ex.command
def preprocess():
  ds = NerDataset()
  ds.preprocess()


@ex.command
def inspect():
  ds = NerDataset()
  inspect_tfrecords(ds.load_train(batch_size=8), max_size=5, visualize_fn=ds.create_print_fn())
  log.info(f"Iterated {iterate_records(ds.load_train(repeat=False)) * ds.batch_size} train records")
  inspect_tfrecords(ds.load_dev(batch_size=2), max_size=5, visualize_fn=ds.create_print_fn())
  log.info(f"Iterated {iterate_records(ds.load_dev()) * ds.batch_size} dev records")
  inspect_tfrecords(ds.load_test(batch_size=2), max_size=5, visualize_fn=ds.create_print_fn())
  log.info(f"Iterated {iterate_records(ds.load_test()) * ds.batch_size} test records")


@ex.command
def validate_dataset():
  import numpy as np
  ds = NerDataset()
  print_fn = ds.create_print_fn()

  slens = []

  def stat_fn(x):
    features, labels = x
    for toks in features['tokens']:
      slens.append(len([t for t in toks if t > 0]))
    # for i in range(len(features['sentence_id'])):
    #   if np.max(labels['boundaries'][i]) > 4:
    #     print_fn(features, labels)
    #     _ = input("Press Enter to view next record...")
    #     break

  iterate_records(ds.load_train(repeat=False, batch_size=32), corpus_stat_fn=stat_fn)
  log.info(f"Avg sent len in train: {sum(slens) / float(len(slens))}")
  slens = []
  iterate_records(ds.load_dev(batch_size=32), corpus_stat_fn=stat_fn)
  log.info(f"Avg sent len in dev: {sum(slens) / float(len(slens))}")
  slens = []
  iterate_records(ds.load_test(batch_size=32), corpus_stat_fn=stat_fn)
  log.info(f"Avg sent len in test: {sum(slens) / float(len(slens))}")


@ex.command
def count_semtypes(_run):
  params = convert_to_namespace(_run.config)
  data_dir = Path(params.dataset.data_dir)
  types = set()

  def get_type_counts(json_dir):
    log.info(f"Counting types in {json_dir}")
    type_counts = defaultdict(lambda: 0)
    for f in json_dir.iterdir():
      doc = json.load(f.open())
      for sentence in doc['sentences']:
        for concept in sentence['concepts']:
          for t in concept['types']:
            type_counts[t] += 1
            types.add(t)
    return type_counts

  train_counts = get_type_counts(data_dir / 'train')
  dev_counts = get_type_counts(data_dir / 'dev')
  test_counts = get_type_counts(data_dir / 'test')

  with (data_dir / 'types.txt').open('w+') as f:
    for t in sorted(types):
      s = "{0}: {1:5d} {2:5d} {3:5d}".format(t, train_counts[t], dev_counts[t], test_counts[t])
      f.write(f"{s}\n")
      log.info(s)


@ex.command
def generate_clef_mentions():
  clef.dump_mentions()


@ex.command
def generate_medmention_mentions():
  medmentions.dump_mentions()


@ex.automain
def main():
  pass
