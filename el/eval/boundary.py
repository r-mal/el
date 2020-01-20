from typing import List, Dict, Generator
from tqdm import tqdm
from hedgedog.logging import get_logger
from hedgedog.nlp.seq_tag import IOBESContinuationBoundaryAggregator

from mm.eval import init_model, f1_eval
from mm.eval.span import Span, wpid_sequence_to_string

log = get_logger("mm.eval.boundary")


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

  for sentence in tqdm(predict_boundaries(model_class, params), total=num_steps):
    sentence_texts[sentence['sentence_id']] = wpid_sequence_to_string(sentence['tokens'], id2wp)
    gold_spans += sentence['gold_spans']
    predicted_spans += sentence['spans']

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


def get_spans(boundaries: List[str], token_ids: List[int], sid: str, id2wp: Dict[int, str], wptokens: List[int]) -> List[Span]:
  boundary_aggregator = IOBESContinuationBoundaryAggregator()
  try:
    for i, (tid, b) in enumerate(zip(token_ids, boundaries)):
      boundary_aggregator.process_item(b, (tid, i))
  except IndexError as e:
    log.error(e)
    log.error(f"exception occurred at item {i} of {boundaries}")
    exit()
  boundary_aggregator.save_span()

  spans = [Span(set(items), sid, id2wp=id2wp, wptokens=wptokens) for items in boundary_aggregator.spans]
  return spans
