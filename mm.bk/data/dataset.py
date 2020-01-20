from pathlib import Path
from typing import Generator
import json
from hedgedog.tf.estimator.ingredients import dataset_ingredient
from hedgedog.tf.io.dataset import FeatureDataset, T
from hedgedog.tf.io.Feature import *
from hedgedog.nlp.wordpiece_tokenization import load_wordpiece_tokenizer
from hedgedog.logging import get_logger
from hedgedog.nlp.spacy.umls import UmlsCandidateGenerator
import numpy as np
from collections import defaultdict

log = get_logger("mm.data.dataset")


class MMDataset(FeatureDataset):
  @dataset_ingredient.capture
  def __init__(self, data_dir: str, batch_size: int, bert_model: str, project_dir: str, candidates_per_concept):
    super().__init__(data_dir, batch_size)
    self.wptokenizer = load_wordpiece_tokenizer(bert_model)
    self.cui2id = json.load((Path(project_dir) / 'info' / 'cui2id.json').open())
    self.umls = UmlsCandidateGenerator()
    self.candidates_per_concept = candidates_per_concept
    # self.type2id = json.load((data_dir / 'type2id.json').open())

    # for umls search accuracy
    self.stats = {
      'train': defaultdict(lambda: 0.),
      'dev': defaultdict(lambda: 0.),
      'test': defaultdict(lambda: 0.)
    }

  def _context_features(self):
    return [
      StringFeature('sentence_id', []),
      IntFeature('num_concepts', [])
    ]

  def _sequence_features(self):
    return [
      IntFeature('tokens', [None]),
      FloatFeature('concept_token_masks', [None, None], post_process=reshape_flat_tensor),
      IntFeature('candidates', [None, None], post_process=reshape_flat_tensor),
      IntFeature('cuis', [None], is_label=True),
      # IntFeature('type_labels', [None, None], is_label=True, post_process=reshape_flat_tensor),
      # IntFeature('boundary_labels', [None], is_label=True)
    ]

  def _get_feature_values(self, sentence):
    wptokens = []
    wpidx_per_token = []
    for t in sentence['tokens']:
      wpts = self.wptokenizer.tokenize_to_id(t['text'])
      wpidx_per_token.append(list(range(len(wptokens), len(wptokens) + len(wpts))))
      wptokens.extend(wpts)

    concepts = [c for c in sentence['concepts'] if c['cui'] in self.cui2id]  # skip concepts without embeddings
    self.stats[sentence['split']]["num_filtered"] += (len(sentence['concepts']) - len(concepts))
    num_concepts = len(concepts)
    concept_token_masks = np.zeros((num_concepts, len(wptokens)), dtype=float)
    cuis = []
    candidates = np.ones((num_concepts, self.candidates_per_concept), dtype=int) * self.cui2id['NO MAP']
    for c, concept in enumerate(concepts):
      # normalization labels
      cuis.append(self.cui2id[concept['cui']])

      # cui candidates
      candidates = self.umls.candidate_generator(concept['text'], self.candidates_per_concept)
      # TODO: evaluate top guess (dev/test sets only)
      if candidates[0].cui == concept['cui']:
        self.stats[sentence['split']]["correct"] += 1.
      self.stats[sentence['split']]["total"] += 1.
      # TODO: store top candidates
      # TODO: count potential candidates (average, median) - aggregate and split by set

      # concept token mapping
      cstart = concept['start']
      cend = concept['end']
      for token, wpidx in zip(sentence['tokens'], wpidx_per_token):
        if token['start'] >= cstart and token['end'] <= cend:
          for w in wpidx:
            concept_token_masks[c, w] = 1.0

    return {
      'sentence_id': sentence['sid'],
      'num_concepts': num_concepts,
      'tokens': wptokens,
      'concept_token_masks': concept_token_masks.flatten(),
      'candidates': candidates.flatten(),
      'cuis': cuis,
    }

  def _example_generator(self, data_file: Path) -> Generator[T, None, None]:
    yield json.load(data_file.open('r'))

  def _example_splitter(self, example: T, split: str) -> Generator[T, None, None]:
    for sentence in example['sentences']:
      sentence['split'] = split
      yield sentence

  def create_print_fn(self):
    id2wp = {v: k for k, v in self.wptokenizer.vocab.items()}
    id2cui = {v: k for k, v in self.cui2id.items()}

    def print_fn(features, labels):
      for i in range(len(features['sentence_id'])):
        tokens = [id2wp[t] for t in features['tokens'][i] if id2wp[t] != '[PAD]']
        log.info("--------------------------------------")
        log.info(f"Sentence ({features['sentence_id'][i]}): {' '.join(tokens)}")
        log.info("-----------------")
        log.info("Concepts:")

        concept_spans = _find_spans(tokens, features['concept_token_masks'][i])
        for concept, cui_id in zip(concept_spans, labels['cuis'][i]):
          log.info(f"-{concept} ({id2cui[cui_id]})")

    return print_fn

  def _finalize_pre_processing(self) -> None:
    correct, total = 0., 0.
    for split in ['train', 'dev', 'test']:
      stats = self.stats[split]
      log.info(split)
      log.info(f"Num filtered: {stats['num_filtered']}/{stats['num_filtered'] + stats['total']}")
      log.info(f"  SciSpacy accuracy: {stats['correct']/stats['total']} ({stats['correct']}/{stats['total']})")


def reshape_flat_tensor(flat_tensor, context_features, sequence_features):
  return tf.reshape(flat_tensor, [context_features['num_concepts'], -1])


def _find_spans(tokens, masks):
  spans = []
  for mask in masks:
    span = ''
    for m, t in zip(mask, tokens):
      if m == 1:
        span = _join_wp_tokens(span, t)
    if len(span) > 1:
      spans.append(span.strip())
  return spans


def _join_wp_tokens(span, new_token):
  sep = ' '
  if new_token[0:2] == '##':
    new_token = new_token[2:]
    sep = ''
  return span + sep + new_token
