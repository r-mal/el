from collections import defaultdict
from typing import Set
from tqdm import tqdm

from el.eval.span import Span


class Evaluation:
  def __init__(self, labels: Set[Span], predictions: Set[Span], partial: bool = False):
    self.tp = []
    self.fp = None
    self.fn = []
    self.match = lambda s1, s2: s1.partial_match(s2) if partial else s1.match(s2)

    sid2preds = defaultdict(lambda: [])
    for s in predictions:
      sid2preds[s.sid].append(s)

    for label in tqdm(labels, desc='Evaluating', total=len(labels)):
      unmatched = True
      matched_pred = None
      for prediction in sid2preds[label.sid]:
        # if prediction in predictions:
        if self.match(label, prediction):
          self.tp += [label]
          unmatched = False
          matched_pred = prediction
          break
      if unmatched:
        self.fn += [label]
      else:
        if matched_pred in predictions:
          predictions.remove(matched_pred)
    self.fp = list(predictions)

  def precision(self):
    return len(self.tp) / max(1., float(len(self.tp) + len(self.fp)))

  def recall(self):
    return len(self.tp) / max(1., float(len(self.tp) + len(self.fn)))

  def f1(self):
    return 2. * self.precision() * self.recall() / max(1., (self.precision() + self.recall()))
