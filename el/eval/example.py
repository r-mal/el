from typing import List

from mm.eval.span import Span


class Example:
  def __init__(self, sid: str, sentence_text: str, tps: List[Span], fps: List[Span], fns: List[Span]):
    self.sid = sid
    self.sentence = sentence_text
    self.tps = [f"({span.cui}): {span.text}" for span in tps]
    self.fps = [f"({span.cui}): {span.text}" for span in fps]
    self.fns = [f"({span.cui}): {span.text}" for span in fns]

  def __str__(self):
    delim = '\n  '
    fp = f"\n  {delim.join(self.fps)}" if len(self.fps) > 0 else ""
    fn = f"\n  {delim.join(self.fns)}" if len(self.fns) > 0 else ""
    tp = f"\n  {delim.join(self.tps)}" if len(self.tps) > 0 else ""
    return f"-----------------------------------------" \
           f"\n Sentence {self.sid}: {self.sentence}" \
           f"\n False Positives: ({len(self.fps)}){fp}" \
           f"\n False Negatives: ({len(self.fns)}){fn}" \
           f"\n True Positives: ({len(self.tps)}){tp}" \
           f"\n-----------------------------------------\n"
