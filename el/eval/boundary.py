from typing import List, Dict, Generator
from tqdm import tqdm
import json
from pathlib import Path
from hedgedog.logging import get_logger
from hedgedog.nlp.seq_tag import IOBESContinuationBoundaryAggregator

from el.eval import init_model, f1_eval
from el.eval.span import Span, wpid_sequence_to_string

log = get_logger("el.eval.boundary")


def boundary_eval(model_class, params):
  """
  Conducts f1/p/r eval given a list of prediction and label dicts
  """
  log.info("Beginning evaluation...")
  ds = model_class.dataset()

  id2wp = {v: k for k, v in ds.wptokenizer.vocab.items()}
  gold_spans = []
  predicted_spans = []
  sentence_texts = {}
  num_steps = params.dataset.num_test if params.estimator.eval_test_set else params.dataset.num_dev
  sentence_dicts = {}
  text2id = {}

  for sentence in tqdm(predict_boundaries(model_class, params), total=num_steps):
    sid = sentence['sentence_id']
    sentence_texts[sid] = wpid_sequence_to_string(sentence['tokens'], id2wp)
    gold_spans += sentence['gold_spans']
    predicted_spans += sentence['spans']
    for span in sentence['spans']:
      if span.text not in text2id:
        text2id[span.text] = len(text2id)

    sentence_dicts[sid] = {
      'sentence_id': sid,
      'num_concepts': len(sentence['spans']),
      'tokens': [int(t) for t in sentence['tokens']],
      'spans': [s.to_dict() for s in sentence['spans']]
    }

  # save predictions such that they can be loaded by entity linking prediction module
  run_name = params.estimator.run_name if params.estimator.boundary_run is None else params.estimator.boundary_run
  split_name = 'test' if params.estimator.eval_test_set else 'dev'
  out_dir = Path(params.estimator.model_dir) / run_name / split_name
  out_dir.mkdir(exist_ok=True)
  log.info(f"Saving sentence predictions to {out_dir}")
  json.dump(sentence_dicts, (out_dir / 'predictions.json').open('w+'))
  json.dump(text2id, (out_dir / 'text2id.json').open('w+'))

  f1_eval(f'boundary', gold_spans, predicted_spans, sentence_texts, params)


def predict_boundaries(model_class, params) -> Generator[Dict[str, None], None, None]:
  run_name = params.estimator.run_name if params.estimator.boundary_run is None else params.estimator.boundary_run
  ckpt = params.estimator.ckpt if params.estimator.boundary_ckpt is None else params.estimator.boundary_ckpt
  estimator, ds, ckpt = init_model(model_class, params, ['boundary'], run_name, ckpt)
  id2boundary = {v: k for k, v in ds.tag2id.items()}
  id2wp = {v: k for k, v in ds.wptokenizer.vocab.items()}

  def dataset_supplier():
    if params.estimator.eval_test_set:
      return ds.load_test()
    else:
      return ds.load_dev()

  for sentence in estimator.predict(dataset_supplier, checkpoint_path=ckpt):
    sid = sentence['sentence_id'].decode('utf-8')
    sentence['sentence_id'] = sid
    preds = sentence['predicted_boundaries']
    sentence['gold_spans'] = get_spans([id2boundary[b] for b in sentence['boundaries']], sentence['token_idx_mapping'],
                                       sid, id2wp, sentence['tokens'])
    sentence['spans'] = get_spans([id2boundary[b] for b in preds], sentence['token_idx_mapping'], sid, id2wp,
                                  sentence['tokens'])

    yield sentence


def get_spans(boundaries: List[str], token_ids: List[int], sid: str, id2wp: Dict[int, str], wptokens: List[int])\
    -> List[Span]:
  boundary_aggregator = IOBESContinuationBoundaryAggregator()
  try:
    for i, (tid, b) in enumerate(zip(token_ids, boundaries)):
      boundary_aggregator.process_item(b, (tid, i))
  except IndexError as e:
    log.error(e)
    # noinspection PyUnboundLocalVariable
    log.error(f"exception occurred at item {i} of {boundaries}")
    exit()
  boundary_aggregator.save_span()

  spans = [Span(set(items), sid, id2wp=id2wp, wptokens=wptokens) for items in boundary_aggregator.spans]
  return spans
