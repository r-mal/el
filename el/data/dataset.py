from pathlib import Path
from typing import Generator
import json
import numpy as np
from collections import defaultdict
from hedgedog.tf.estimator.ingredients import dataset_ingredient
from hedgedog.tf.io.dataset import FeatureDataset, T
from hedgedog.tf.io.Feature import *
from hedgedog.nlp.wordpiece_tokenization import load_wordpiece_tokenizer
from hedgedog.logging import get_logger
from hedgedog.nlp.spacy.umls import UmlsCandidateGenerator
from hedgedog.nlp.seq_tag import get_tags, ContinuationBoundaryLabeler, IOBESContinuationBoundaryAggregator

log = get_logger("mm.data.dataset")


class NerDataset(FeatureDataset):
  @dataset_ingredient.capture
  def __init__(self, data_dir: str, batch_size: int, bert_model: str, project_dir: str, candidates_per_concept: int,
               record_dir_name: str, tagset, ignore_sentences_without_concepts, dataset, mention_candidate_path):
    super().__init__(data_dir, batch_size)
    info_dir = Path(project_dir) / 'info'
    self.wptokenizer = load_wordpiece_tokenizer(bert_model)
    self.cui2id = json.load((info_dir / 'cui2id.json').open())
    self.tui2label_id = json.load((info_dir / 'tui2label.json').open())
    self.candidates_per_concept = candidates_per_concept
    self.filter = ignore_sentences_without_concepts
    self.mention2idx = json.load((info_dir / f'{dataset}_mentions.json').open())

    # for umls search accuracy
    self.stats = {
      'train': defaultdict(lambda: 0.),
      'dev': defaultdict(lambda: 0.),
      'test': defaultdict(lambda: 0.)
    }
    self.record_dir_name = record_dir_name
    self.tags = get_tags(tagset)
    self.tag2id = self.tags.tag2id()
    self._umls = None
    if mention_candidate_path is not None:
      with np.load(mention_candidate_path) as npz:
        self.mention_top_distances = npz['distances']
        self.mention_top_candidates = npz['candidates']
        id2cui = {i: c for (c, i) in self.cui2id.items()}
        mention_candidates = {}
        for m_idx, (m_cids, m_cdists) in enumerate(zip(self.mention_top_candidates, self.mention_top_distances)):
          mention_candidates[m_idx] = []
          for m_cid, m_cdist in zip(m_cids, m_cdists):
            candidate = {
              'code': id2cui[m_cid],
              'score': (2.0 - (0.5 * m_cdist))
            }
            mention_candidates[m_idx].append(candidate)
        self.mention_candidates = mention_candidates
    else:
      self.mention_candidates = None
    log.info(f"Initialized dataset with tags {tagset} at {data_dir}")

  @property
  def umls(self):
    """ Lazy load umls """
    if self._umls is None:
      self._umls = UmlsCandidateGenerator()
    return self._umls

  def _context_features(self):
    return [
      StringFeature('sentence_id', []),
      IntFeature('num_concepts', [])
    ]

  def _sequence_features(self):
    return [
      IntFeature('tokens', [None]),
      FloatFeature('concept_token_masks', [None, None], post_process=reshape_flat_tensor),
      IntFeature('candidates', [None, None], is_label=True, post_process=fix_candidates),
      FloatFeature('candidate_scores', [None, None], is_label=True, post_process=reshape_flat_tensor),
      FloatFeature('candidate_mask', [None, None], is_label=True, post_process=reshape_flat_tensor),
      FloatFeature('gold_in_candidate_mask', [None], is_label=True),
      IntFeature('boundaries', [None], is_label=True),
      IntFeature('concept_head_idx', [None]),
      IntFeature('token_idx_mapping', [None], is_label=True),
      IntFeature('semantic_types', [None, None], post_process=reshape_flat_tensor),
      IntFeature('semtype_labels', [None, None], is_label=True, post_process=reshape_flat_tensor),
      IntFeature('mention_embedding_idx', [None])
    ]

  def _get_feature_values(self, sentence):
    wptokens = []
    wpidx_per_token = []
    token_idx_mapping = []
    for i, t in enumerate(sentence['tokens']):
      wpts = self.wptokenizer.tokenize_to_id(t['text'])
      wpidx_per_token.append(list(range(len(wptokens), len(wptokens) + len(wpts))))
      wptokens.extend(wpts)
      token_idx_mapping += [i] * len(wpts)

    concepts = [c for c in sentence['concepts'] if self._valid_concept(c)]  # skip concepts without embeddings
    split = sentence['split']
    self.stats[split]["num_filtered"] += (len(sentence['concepts']) - len(concepts))
    real_num_concepts = len(concepts)
    num_concepts = max(1, len(concepts))  # we do this to avoid loading tensors with 0's in their shape
    concept_token_masks = np.zeros((num_concepts, len(wptokens)), dtype=float)
    candidates_np = np.ones((num_concepts, self.candidates_per_concept), dtype=int) * self.cui2id['CUI-less']
    candidate_mask = np.zeros((num_concepts, self.candidates_per_concept), dtype=float)
    candidate_scores = np.zeros((num_concepts, self.candidates_per_concept), dtype=float)
    gold_in_candidate_mask = np.zeros([num_concepts], dtype=float)
    semantic_types = np.zeros((num_concepts, 4), dtype=int)
    type_labels = np.zeros((num_concepts, len(self.tui2label_id)), dtype=int)
    mention_embedding_idx = []

    # prepare boundary labels
    boundary_labels = [self.tag2id[self.tags.outside()] for _ in wptokens]
    concept_head_idx = []

    for c, concept in enumerate(concepts):
      self.stats[split]["total"] += 1.

      if self.mention_candidates is not None:
        candidate_list = self.mention_candidates[concept['embedding']]
      else:
        candidate_list = concept['candidates']
      # normalization labels
      candidates = [self.cui2id[concept['cui']]]
      scores = [0.]
      gold_cui_rank = -1
      gold_cui_in_candidates = False
      for idx, candidate in enumerate(candidate_list):
        if candidate['code'] == concept['cui']:
          scores[0] = candidate['score']
          gold_cui_in_candidates = True
          self.stats[split]["gold_rank_sum"] += idx + 1
          self.stats[split]["gold_rank_count"] += 1
        elif candidate['code'] in self.cui2id:
          candidates.append(self.cui2id[candidate['code']])
          scores.append(candidate['score'])
        # should not count gold if not in list we will use
        if len(candidates) >= self.candidates_per_concept:
          break
      best_guess = candidate_list[0]['code'] if len(candidate_list) > 0 else 'CUI-less'
      # trim candidates/scores
      candidates = candidates[:self.candidates_per_concept]
      scores = scores[:self.candidates_per_concept]
      candidate_scores[c, :len(scores)] = scores

      # pad candidates with random cuis
      if len(candidates) < self.candidates_per_concept:
        num_random = self.candidates_per_concept - len(candidates)
        candidates.extend(np.random.choice(len(self.cui2id), num_random, replace=False))
        # TODO: scores for random candidates?

      # add in CUI-less candidate
      if concept['cui'] != 'CUI-less':
        candidates[-1] = self.cui2id['CUI-less']
        candidate_scores[c, len(candidates)-1] = 0.

      # cui candidates
      candidates_np[c, :len(candidates)] = candidates
      candidate_mask[c, :len(candidates)] = [1.] * len(candidates)
      
      # evaluate top guess (dev/test sets only)
      if best_guess == concept['cui']:
        self.stats[split]["correct"] += 1.

      # count potential candidates
      self.stats[split]['num_candidates'] += len(candidates)
      if gold_cui_in_candidates or concept['cui'] == 'CUI-less':
        self.stats[split]['gold_in_candidates'] += 1.
        gold_in_candidate_mask[c] = 1.

      # concept token mapping
      for token, wpidx in zip(sentence['tokens'], wpidx_per_token):
        for span in concept['spans']:
          if overlap(token['start'], token['end'], span['start'], span['end']):
            for w in wpidx:
              concept_token_masks[c, w] = 1.0

      # boundary labeling
      # combine the boundary labels for each concept mention into a single sequence
      collision = False
      for i, bl in enumerate(self._create_boundary_labels(concept_token_masks[c])):
        if bl in {self.tag2id[self.tags.begin()], self.tag2id[self.tags.single()]}:
          concept_head_idx.append(i)
        if boundary_labels[i] == 0:
          boundary_labels[i] = bl
        elif bl != 0:
          collision = True
      if collision:
        self.stats[split]['boundary_label_collision'] += 1

      # type info
      for idx, t in enumerate(concept['types']):
        semantic_types[c, idx] = self.cui2id[t]
        type_labels[c, self.tui2label_id[t]] = 1

      # mention idx
      mention_idx = concept['embedding']

      mention_embedding_idx.append(mention_idx)


    # B = self.tag2id[self.tags.begin()]

    assert len(concept_head_idx) == real_num_concepts, f'{sentence["sid"]} Does not match span counts: ' \
                                                       f'{len(concept_head_idx)} vs {real_num_concepts}'

    # add cls and sep tokens
    wptokens = [self.wptokenizer.vocab['[CLS]']] + wptokens + [self.wptokenizer.vocab['[SEP]']]
    # add empty cls and sep masks
    concept_token_masks = np.concatenate(
      [
        np.zeros((num_concepts, 1), dtype=float),
        concept_token_masks,
        np.zeros((num_concepts, 1), dtype=float)
      ],
      axis=-1
    )
    # add none cls and sep boundary labels
    boundary_labels = [self.tag2id[self.tags.outside()]] + boundary_labels + [self.tag2id[self.tags.outside()]]
    # add 1 for cls
    concept_head_idx = [i+1 for i in concept_head_idx]
    # add 1 for cls, add cls and sep
    token_idx_mapping = [-1] + token_idx_mapping + [-1]

    return {
      'sentence_id': sentence['sid'],
      'num_concepts': real_num_concepts,
      'tokens': wptokens,
      'concept_token_masks': concept_token_masks.flatten(),
      'candidates': candidates_np.flatten(),
      'candidate_mask': candidate_mask.flatten(),
      'candidate_scores': candidate_scores.flatten(),
      'gold_in_candidate_mask': gold_in_candidate_mask,
      'boundaries': boundary_labels,
      'concept_head_idx': concept_head_idx,
      'token_idx_mapping': token_idx_mapping,
      'semantic_types': semantic_types.flatten(),
      'semtype_labels': type_labels.flatten(),
      'mention_embedding_idx': mention_embedding_idx
    }

  def _create_boundary_labels(self, token_mask):
    boundary_labels = []
    boundary_labeler = ContinuationBoundaryLabeler(self.tags, lambda x: x > 0.)
    for i, flag in enumerate(token_mask):
      next_flag = 0. if i+1 >= len(token_mask) else token_mask[i+1]
      boundary_labels.append(self.tag2id[boundary_labeler.label(flag, next_flag)])
    return boundary_labels

  def _example_generator(self, data_file: Path) -> Generator[T, None, None]:
    yield json.load(data_file.open('r'))

  def _example_splitter(self, example: T, split: str) -> Generator[T, None, None]:
    for sentence in example['sentences']:
      # log.info(f"Processing sentence {sentence['sid']}")
      sentence['split'] = split
      yield sentence

  def _valid_concept(self, concept):
    if concept['cui'] == 'NO MAP':
      concept['cui'] = 'CUI-less'
    return concept['cui'] in self.cui2id

  def create_print_fn(self):
    self.umls('disease', 1)  # prepare umls
    id2wp = {v: k for k, v in self.wptokenizer.vocab.items()}
    id2cui = {v: k for k, v in self.cui2id.items()}
    id2tag = {v: k for k, v in self.tag2id.items()}
    id2tui = {v: k for k, v in self.tui2label_id.items()}

    def print_fn(features, labels):
      for i in range(len(features['sentence_id'])):
        tokens = [id2wp[t] for t in features['tokens'][i] if id2wp[t] != '[PAD]']
        log.info("--------------------------------------")
        log.info(f"Sentence ({features['sentence_id'][i]}): {wp_sequence_to_string(tokens)}")
        tandb = [f"({t}|{id2tag[b]})" for t, b in zip(tokens, features['boundaries'][i])]
        log.info(f"Tokens: {' '.join(tandb)}({len(features['tokens'][i])})")
        log.info("-----------------")
        log.info("Concepts:")
        concept_spans = _find_spans(tokens, features['concept_token_masks'][i])
        for concept, candidates, scores, mask, semtypes in zip(concept_spans,
                                                               labels['candidates'][i],
                                                               labels['candidate_scores'][i],
                                                               labels['candidate_mask'][i],
                                                               # labels['semtype_labels'][i],
                                                               labels['candidate_mask'][i]):
          cui = id2cui[candidates[0]]
          types = []  # [id2tui[i] for i, t in enumerate(semtypes) if t > 0]
          name = self.umls.get_concept_entity(cui).canonical_name
          candidates = [(id2cui[c], "{0:.3f}".format(s)) for c, s, m in zip(candidates, scores, mask) if m > 0.]
          candidates = [f"({s}|{c}|{'' if c == 'CUI-less' else self.umls.get_concept_entity(c).canonical_name})"
                        for c, s in candidates]
          log.info(f" {concept} ({cui}|{name}|{types}). Candidates ({len(candidates)}): {candidates[:5]} ... {candidates[-5:]}")

        log.info("-----------------")
        log.info("Spans recovered from boundary labels:")
        aggregator = IOBESContinuationBoundaryAggregator()
        aggregator.process_sequence([id2tag[b] for b in features['boundaries'][i][:len(tokens)]], tokens)
        for sp in aggregator.spans:
          log.info(f"  {wp_sequence_to_string(sp)}")

    return print_fn

  def _finalize_pre_processing(self) -> None:
    correct, total, filtered, candidates, found_gold = 0., 0., 0, 0., 0.
    for split in ['train', 'dev', 'test']:
      stats = self.stats[split]
      log.info("---------------")
      log.info(split)
      log.info(f"  CUIs without embeddings filtered: {stats['num_filtered']}/{stats['num_filtered'] + stats['total']}")
      log.info(f"  SciSpacy accuracy: {stats['correct']/stats['total']} ({stats['correct']}/{stats['total']})")
      log.info(f"  Avg number of candidates per cui: {stats['num_candidates']/stats['total']}")
      log.info(f"  Gold CUIs found in search: {stats['gold_in_candidates']/stats['total']}"
               f" ({stats['gold_in_candidates']}/{stats['total']})")
      g_mean_rank = stats['gold_rank_sum'] / stats['gold_rank_count']
      log.info(f"  Gold Mean Rank: {g_mean_rank}")

      log.info(f"  Boundary label collisions: {stats['boundary_label_collision']/stats['total']}"
               f" ({stats['boundary_label_collision']}/{stats['total']})")
      correct += stats['correct']
      total += stats['total']
      filtered += stats['num_filtered']
      candidates += stats['num_candidates']
      found_gold += stats['gold_in_candidates']
    log.info("---------------")
    log.info("All")
    log.info(f"  CUIs without embeddings filtered: {filtered}/{filtered + total} ({filtered/(filtered + total)})")
    log.info(f"  SciSpacy accuracy: {correct/total} ({correct}/{total})")
    log.info(f"  Avg number of candidates per cui: {candidates/total}")
    log.info(f"  Gold CUIs found in search: {found_gold/total} ({found_gold}/{total})")
    json.dump(self.stats, (Path(self.data_dir) / self.record_dir_name / 'dataset_stats.json').open('w+'))

  def _filter(self, features, labels):
    # return tf.logical_and(tf.math.equal(tf.strings.substr(features['sentence_id'], 0,
    #                                                       len('01314-028800-DISCHARGE_SUMMARY')),
    #                                     b'01314-028800-DISCHARGE_SUMMARY'),
    #                       tf.greater(features['num_concepts'], 0))
    return (not self.filter) or tf.greater(features['num_concepts'], 0)


# noinspection PyUnusedLocal
def reshape_flat_tensor(flat_tensor, context_features, sequence_features):
  return tf.reshape(flat_tensor, [tf.maximum(context_features['num_concepts'], 1), -1])


def fix_candidates(flat_tensor, context_features, sequence_features):
  reshaped = tf.reshape(flat_tensor, [tf.maximum(context_features['num_concepts'], 1), -1])
  # replace 3210963 (CUI-less) with 3210782 (TOO1)
  idx = tf.cast(tf.equal(reshaped, 3210963), tf.int64)
  same_mask = tf.cast(tf.not_equal(reshaped, 3210963), tf.int64)
  new_val = idx * 3210782

  return reshaped * same_mask + new_val


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


def wp_sequence_to_string(wp_tokens):
  span = ''
  for t in wp_tokens:
    span = _join_wp_tokens(span, t)
  return span


def _join_wp_tokens(span, new_token):
  sep = ' '
  if new_token[0:2] == '##':
    new_token = new_token[2:]
    sep = ''
  return span + sep + new_token


def overlap(s1, e1, s2, e2):
  return not (s1 >= e2 or s2 >= e1)
