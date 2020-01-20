from pathlib import Path
from typing import List
from string import punctuation
from hedgedog.logging import get_logger
from hedgedog.nlp.spacy import SpacyAnnotator
from hedgedog.nlp.spacy.umls import UmlsCandidate, UmlsCandidateGenerator

log = get_logger("mm.data.brat")

punctuation = punctuation.replace('\'', '')


class LazySpacy:
  def __init__(self):
    self.spacy = None

  def annotate(self, text):
    if self.spacy is None:
      self.spacy = SpacyAnnotator('en_core_sci_sm', 'default', ['parser'])
    return self.spacy.annotate(text)


spacy = LazySpacy()


class Span:
  def __init__(self, start: int, end: int, text: str):
    self.start = start
    self.end = end
    self.text = text

  def to_brat(self, eid, note=None):
    brat_string = f"T{eid}\tConcept {self.start} {self.end}\t{self.text}"
    if note is not None:
      brat_string += f"\n#{eid}\tAnnotatorNotes T{eid}\t{note}"
    return brat_string

  def to_json(self):
    return {
      'start': self.start,
      'end': self.end,
      'text': self.text
    }


class Concept:
  def __init__(self, spans: List[Span], types: List[str], cui: str, candidates: List[UmlsCandidate] = None,
               embedding_idx: int = None):
    self.spans = spans
    self.types = types
    self.cui = cui
    self.candidates = candidates
    self.embedding_idx = embedding_idx

  def to_brat(self, eid, rid=None):
    note = self.cui + '--' + ','.join(self.types)
    brat_string = self.spans[0].to_brat(eid, note)
    orig_eid = eid
    if len(self.spans) > 1:
      for span in self.spans[1:]:
        eid += 1
        brat_string += '\n' + span.to_brat(eid)
        brat_string += f"\nR{rid}\tCONTINUATION Arg1:T{orig_eid} Arg2:T{eid}"
        rid += 1
    return eid + 1, rid, brat_string

  def to_json(self):
    return {
      'spans': [s.to_json() for s in self.spans],
      'types': self.types,
      'cui': self.cui,
      'candidates': [c.to_json() for c in self.candidates],
      'embedding': self.embedding_idx
    }

  @property
  def start(self):
    return min(s.start for s in self.spans)

  @property
  def end(self):
    return max(s.end for s in self.spans)

  def text(self):
    return ' '.join(s.text for s in self.spans)


class Token:
  def __init__(self, start: int, text: str):
    self.start = start
    self.text = text

  def to_json(self):
    return {
      'start': self.start,
      'end': self.start + len(self.text),
      'text': self.text,
    }


class Sentence:
  def __init__(self, sid: str, tokens: List, all_concepts: List[Concept], doc_text):
    self.sid = sid
    sentence_start = tokens[0].idx
    sentence_end = tokens[-1].idx + len(tokens[-1])
    assert doc_text[sentence_start:sentence_end].strip() == ''.join(t.string for t in tokens).strip(),\
        f"({sentence_start},{sentence_end})|{doc_text[sentence_start:sentence_end].strip()}| != \n" \
        f"|{''.join(t.string for t in tokens).strip()}|"

    self.local_concepts = []
    for concept in all_concepts:
      if concept.start >= sentence_start and concept.end <= sentence_end:
        adjusted_spans = [Span(s.start - sentence_start, s.end - sentence_start, s.text) for s in concept.spans]
        self.local_concepts.append(Concept(adjusted_spans, concept.types, concept.cui, concept.candidates,
                                           concept.embedding_idx))
    self.tokens = [t_ for t in tokens for t_ in create_tokens(t, sentence_start)]

  def to_json(self):
    return {
      'sid': self.sid,
      'concepts': [c.to_json() for c in self.local_concepts],
      'tokens': [t.to_json() for t in self.tokens]
    }


def create_tokens(spacy_token, offset):
  start = spacy_token.idx - offset
  tokens = []
  token_chars = []
  for c in spacy_token.text:
    if c in punctuation:
      if len(token_chars) > 0:
        tokens.append(Token(start, ''.join(token_chars)))
        start += len(token_chars)
        token_chars = []
      tokens.append(Token(start, c))
      start += 1
    else:
      token_chars.append(c)
  if len(token_chars) > 0:
    tokens.append(Token(start, ''.join(token_chars)))
  return tokens


class Document:
  def __init__(self, did: str, text: str, concepts: List[Concept], umls: UmlsCandidateGenerator, k, mention2idx):
    self.did = did
    self.text = text
    self.concepts = concepts
    self.sentences = []

    # add concept embedding to each concept
    for concept in self.concepts:
      concept.embedding_idx = mention2idx[concept.text().strip()]

    # perform cui search for each concept
    batch_results = umls.batch_search([c.text() for c in self.concepts], k + 20)
    for concept, results in zip(self.concepts, batch_results):
      concept.candidates = results

    # create sentences
    spacy_doc = spacy.annotate(text)
    for sid, span in enumerate(spacy_doc.sents):
      self.sentences.append(Sentence(f"{self.did}S{sid}", spacy_doc[span.start: span.end], self.concepts, text))

  def to_brat(self, outdir: Path):
    textfile = outdir / f"{self.did}.txt"
    with textfile.open('w+') as f:
      f.write(self.text)

    entity_count = 0
    relation_count = 0
    annfile = outdir / f"{self.did}.ann"
    with annfile.open('w+') as f:
      for entity in self.concepts:
        entity_count, relation_count, brat_string = entity.to_brat(entity_count, relation_count)
        f.write(brat_string)
        f.write('\n')

  def to_json(self):
    return {
      'did': self.did,
      'sentences': [s.to_json() for s in self.sentences]
    }
