from collections import defaultdict
from pathlib import Path
import tensorflow_estimator as tfe
from hedgedog.logging import get_logger
from hedgedog.tf.estimator import train as hdtrain

from mm.eval.evaluation import Evaluation
from mm.eval.example import Example
from mm.eval.span import Span

log = get_logger("mm.eval")


def f1_eval(name, gold_spans, predicted_spans, sentence_texts, params):
  exact_eval = Evaluation(set(gold_spans), set(predicted_spans))
  report = f"\nExact Boundary Evaluation\n----------" +\
           f"\nP:  {exact_eval.precision()}" +\
           f"\nR:  {exact_eval.recall()}" +\
           f"\nF1: {exact_eval.f1()}"
  log.info(report)

  partial_eval = Evaluation(set(gold_spans), set(predicted_spans), partial=True)
  partial_report = f"\n----------\nPartial Boundary Evaluation\n----------" +\
                   f"\nP:  {partial_eval.precision()}" +\
                   f"\nR:  {partial_eval.recall()}" +\
                   f"\nF1: {partial_eval.f1()}"
  log.info(partial_report)
  report += partial_report

  outdir = Path(params.estimator.model_dir) / params.estimator.run_name / ('test' if params.estimator.eval_test_set else 'dev')
  outdir.mkdir(exist_ok=True)
  with (outdir / f'{name}_eval.txt').open('w+') as f:
    f.write(report)

  tps = defaultdict(list)
  fps = defaultdict(list)
  fns = defaultdict(list)
  for s in exact_eval.tp:
    tps[s.sid].append(s)
  for s in exact_eval.fp:
    fps[s.sid].append(s)
  for s in exact_eval.fn:
    fns[s.sid].append(s)
  log.info(f"{len(fns) + len(fps)} of {len(tps) + len(fns) + len(fps)} sentences with spans contain errors")
  with (outdir / f'{name}_examples.txt').open('w+') as f:
    for sid in list(fps.keys()) + list(fns.keys()):
      try:
        ex = Example(sid, sentence_texts[sid], tps[sid], fps[sid], fns[sid])
      except KeyError as e:
        log.error(e)
        log.error(f"sentence_texts.keys(): {sorted(sentence_texts.keys())}")
        exit()
      f.write(str(ex))


def init_model(model_class, params, modules, run_name, ckpt):
  ds = model_class.dataset()
  model_fn = hdtrain.build_model_fn(model_class, dataset=ds)
  model_dir = str(Path(params.estimator.model_dir) / run_name)
  params.estimator.run_name = run_name
  params.model.modules = modules
  estimator = tfe.estimator.Estimator(model_fn,
                                      model_dir=model_dir,
                                      params=params)
  ckpt = _init_ckpt(ckpt, model_dir)

  return estimator, ds, ckpt


def _init_ckpt(ckpt_num, model_dir):
  ckpt = None
  if ckpt_num is not None:
    ckpt = str(Path(model_dir) / f"model.ckpt-{ckpt_num}")
    log.info(f"Attempting to load ckpt at {ckpt}")
  return ckpt
