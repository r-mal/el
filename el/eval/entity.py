import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Callable, Generator
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from hedgedog.logging import get_logger
from hedgedog.nlp.spacy.umls import UmlsCandidateGenerator
from el.data.dataset import overlap
from el.eval import init_model, f1_eval
from el.eval.span import Span, wpid_sequence_to_string

log = get_logger("el.eval.entity")


def end_to_end_eval(model_class, params):
  gold_spans = _load_gold_spans(params)
  sentence_dicts = _load_sentences(params)
  # num_steps = params.dataset.num_test if params.estimator.eval_test_set else params.dataset.num_dev
  # sentence_dicts: Dict[str, Dict[str, None]] = {s['sentence_id']: s
  #                                            for s in tqdm(predict_boundaries(model_class, params), total=num_steps)}

  log.info("Boundary detection done. Running entity linking...")
  predicted_spans = []
  sentence_texts = {}
  id2wp = {v: k for k, v in model_class.dataset().wptokenizer.vocab.items()}
  for sentence in predict_entities(model_class, params, sentence_dicts):
    # noinspection PyTypeChecker
    predicted_spans.extend(sentence['spans'])
    # noinspection PyTypeChecker
    sentence_texts[sentence['sentence_id']] = wpid_sequence_to_string(sentence['tokens'], id2wp)

  f1_eval('entity', gold_spans, predicted_spans, sentence_texts, params)


def predict_entities(model_class, params, sentences: Dict[str, Dict[str, None]]) \
   -> Generator[Dict[str, None], None, None]:
  run_name = params.estimator.run_name if params.estimator.norm_run is None else params.estimator.norm_run
  ckpt = params.estimator.ckpt if params.estimator.norm_ckpt is None else params.estimator.norm_ckpt

  # load offline embedding stuff
  split_name = 'test' if params.estimator.eval_test_set else 'dev'
  emb_dir = Path(params.estimator.model_dir) / params.estimator.boundary_run / split_name
  emb_idx = json.load((emb_dir / 'text2id.json').open())
  params.model.offline_emb_file = str(emb_dir / 'text_embeddings.npz')

  # init model
  estimator, ds, ckpt = init_model(model_class, params, ['norm'], run_name, ckpt)
  id2code = {v: k for k, v in ds.cui2id.items()}

  dataset_supplier = el_dataset(sentences.values(), params.dataset.candidates_per_concept, ds.cui2id, emb_idx)

  log.info("Performing second pass for entity linking...")
  for sentence in tqdm(estimator.predict(dataset_supplier, checkpoint_path=ckpt), desc='entity', total=len(sentences)):
    sid = sentence['sentence_id'].decode('utf-8')
    sentence['sentence_id'] = sid
    # noinspection PyTypeChecker
    spans: List[Span] = sentences[sid]['spans']
    assert len(spans) == sentence['num_concepts'], \
        log.error(f"Spans: {len(spans)}, num_concepts: {sentence['num_concepts']}")

    for span, candidates, candidate_scores in zip(spans, sentence['candidates'], sentence['candidate_scores']):
      if max(candidate_scores) > 0:
        top_candidate = candidates[np.argmax(candidate_scores)]
        span.cui = id2code[top_candidate]
        top_10_idx = np.argpartition(candidate_scores, -10)[-10:]
        top_10_cuis = ["({0}, {1:.4e})".format(id2code[candidates[i]], candidate_scores[i])
                       for i in sorted(top_10_idx, key=lambda x: candidate_scores[x], reverse=True)]
        span.text = span.text + ' ' + str(top_10_cuis)
    sentence['spans'] = spans
    yield sentence


def el_dataset(sentences, k, code2id, emb_idx) -> Callable[[None], tf.data.Dataset]:
  output_types = {
    'sentence_id': tf.string,
    'num_concepts': tf.int32,
    'tokens': tf.int32,
    'concept_token_masks': tf.float32,
    'candidates': tf.int32,
    'candidate_mask': tf.float32,
    'candidate_scores': tf.float32,
    'mention_embedding_idx': tf.int32
  }

  padded_shapes = {
    'sentence_id': [],
    'num_concepts': [],
    'tokens': [None],
    'concept_token_masks': [None, None],
    'candidates': [None, None],
    'candidate_mask': [None, None],
    'candidate_scores': [None, None],
    'mention_embedding_idx': [None]
  }

  # perform cui search outside of the generator
  cui_less_code = code2id['CUI-less']  # code2id['T001']
  candidate_map = defaultdict(lambda: [(cui_less_code, 0.)])
  umls = UmlsCandidateGenerator()
  for sentence in tqdm(sentences, desc='umls search', total=len(sentences)):
    spans = [s.text for s in sentence['spans'] if s.text not in candidate_map]
    batch_candidates = umls.batch_search(spans, 2*k)
    for sp, candidates in zip(spans, batch_candidates):
      candidate_map[sp] = [(code2id[c.code], c.score)
                           for c in candidates if c.code in code2id][:k-1] + [(cui_less_code, 0.)]

  def sentence_generator():
    for sent in sentences:
      num_concepts = len(sent['spans'])
      concept_token_masks = np.zeros([num_concepts, len(sent['tokens'])], dtype=float)
      cands = np.ones([num_concepts, k], dtype=int) * cui_less_code
      candidate_mask = np.zeros([num_concepts, k], dtype=float)
      candidate_scores = np.zeros([num_concepts, k], dtype=float)
      mention_embedding_idx = []
      for c, span_ in enumerate(sent['spans']):
        # create mask from span.raw_token_ids
        for i in span_.wp_token_ids:
          concept_token_masks[c, i] = 1.
        concept_cands, concept_scores = zip(*candidate_map[span_.text])
        cands[c, :len(concept_cands)] = concept_cands
        candidate_mask[c, :len(concept_cands)] = [1.] * len(concept_cands)
        candidate_scores[c, :len(concept_cands)] = concept_scores
        mention_embedding_idx.append(emb_idx[span_.text])
      yield {
        'sentence_id': sent['sentence_id'],
        'num_concepts': num_concepts,
        'tokens': sent['tokens'],
        'concept_token_masks': concept_token_masks,
        'candidates': cands,
        'candidate_mask': candidate_mask,
        'candidate_scores': candidate_scores,
        'mention_embedding_idx': mention_embedding_idx
      }

  def dataset_supplier():
    return tf.data.Dataset.from_generator(sentence_generator,
                                          output_types=output_types) \
      .padded_batch(batch_size=16, padded_shapes=padded_shapes)

  return dataset_supplier


def _load_gold_spans(params):
  gold_spans = []
  jsondir = Path(params.dataset.data_dir) / ('test' if params.estimator.eval_test_set else 'dev')
  for jfile in jsondir.iterdir():
    doc = json.load(jfile.open())
    for jsentence in doc['sentences']:
      sid = jsentence['sid']
      for jconcept in jsentence['concepts']:
        token_ids = []
        cui = jconcept['cui']
        cui = "CUI-less" if cui == 'NO MAP' else cui
        span_text = ''
        for jspan in jconcept['spans']:
          span_text += ' ' + jspan['text']
          start, end = jspan['start'], jspan['end']
          for i, jtok in enumerate(jsentence['tokens']):
            if overlap(jtok['start'], jtok['end'], start, end):
              token_ids.append(i)
        # noinspection PyUnboundLocalVariable
        assert len(token_ids) > 0, log.error(f"Bad concept in sentence {sid}: ({jspan['start'], jspan['end']})")
        gold_spans.append(Span([(t, t) for t in token_ids], sid, text=span_text.strip(), cui=cui))

  log.info("Gold spans loaded. Running boundary detection...")
  return gold_spans


def _load_sentences(params):
  split_name = 'test' if params.estimator.eval_test_set else 'dev'
  inf = Path(params.estimator.model_dir) / params.estimator.boundary_run / split_name / 'predictions.json'
  sentences = json.load(inf.open())
  for s in sentences.values():
    s['spans'] = [Span.from_dict(sp) for sp in s['spans']]
  return sentences
