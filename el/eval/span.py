from typing import Set, Tuple, List, Dict

from mm.data.dataset import wp_sequence_to_string


class Span:
  def __init__(self, tokens_ids: Set[Tuple[int, int]], sid: str, id2wp: Dict[int, str]=None, wptokens=None, text=None, cui=None):
    self.token_ids, self.wp_token_ids = zip(*tokens_ids)
    self.token_ids = set(self.token_ids)
    self.sid = sid
    if text is None:
      self.text = wpid_sequence_to_string([wptokens[w] for w in sorted(self.wp_token_ids)], id2wp)
    else:
      self.text = text
    self.cui = 'CUI-less' if cui is None else cui

  def match(self, other_span):
    return self.sid == other_span.sid and \
           self.cui == other_span.cui and \
           self.token_ids <= other_span.token_ids <= self.token_ids

  def partial_match(self, other_span):
    return self.sid == other_span.sid and \
           self.cui == other_span.cui and \
           len(self.token_ids & other_span.token_ids) > 0

  def __eq__(self, other):
    return self.match(other)

  def __hash__(self):
    return hash((frozenset(self.token_ids), self.sid))


def wpid_sequence_to_string(token_ids: List[int], id2wp: Dict[int, str], span: Span = None) -> str:
  start = 0 if span is None else min(span.wp_token_ids)
  end = len(token_ids) if span is None else max(span.wp_token_ids) + 1
  return wp_sequence_to_string([id2wp[i] for i in token_ids[start: end] if id2wp[i] != '[PAD]'])