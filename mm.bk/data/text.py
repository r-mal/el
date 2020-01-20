from typing import List
from pathlib import Path
import traceback
from hedgedog.logging import get_logger
from hedgedog.nlp.spacy import SpacyAnnotator
from hedgedog.tf.estimator.ingredients import dataset_ingredient

log = get_logger("mm.data.brat")

spacy = SpacyAnnotator('en_core_sci_sm', 'default', ['parser'])


class Concept:
  def __init__(self, start: int, end: int, text: str, types: List[str], cui: str):
    self.start = start
    self.end = end
    self.text = text
    self.types = types
    self.cui = cui

  def to_brat(self, eid):
    note = self.cui + '--' + ','.join(self.types)
    return f"T{eid}\tConcept {self.start} {self.end}\t{self.text}" +\
        f"\n#{eid}\tAnnotatorNotes T{eid}\t{note}"

  def to_json(self):
    return {
      'start': self.start,
      'end': self.end,
      'text': self.text,
      'types': self.types,
      'cui': self.cui
    }


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
        self.local_concepts.append(
          Concept(concept.start - sentence_start,
                  concept.end - sentence_start,
                  concept.text,
                  concept.types,
                  concept.cui))
    self.tokens = [Token(t.idx - sentence_start, t.text) for t in tokens]

  def to_json(self):
    return {
      'sid': self.sid,
      'concepts': [c.to_json() for c in self.local_concepts],
      'tokens': [t.to_json() for t in self.tokens]
    }


class Document:
  def __init__(self, lines: List[str]):
    self.did, _, self.title = lines[0].strip().split('|')
    self.text = '|'.join(lines[1].strip().split('|')[2:])
    self.entities = []
    for eid, line in enumerate(lines[2:]):
      did, start, end, text, types, cui = line.strip().split('\t')
      start, end = int(start), int(end)
      if start > len(self.title):
        start += 1
        end += 1
      self.entities.append(Concept(start, end, text, types.split(','), cui))
    self.title += '.'

    self.sentences = []
    doc_string = self.title + ' ' + self.text
    spacy_doc = spacy.annotate(doc_string)
    for sid, span in enumerate(spacy_doc.sents):
      self.sentences.append(Sentence(f"{self.did}S{sid}", spacy_doc[span.start: span.end], self.entities, doc_string))

  def to_brat(self, outdir: Path):
    textfile = outdir / f"{self.did}.txt"
    with textfile.open('w+') as f:
      f.write(self.title)
      f.write('\n')
      f.write(self.text)

    entity_count = 0
    annfile = outdir / f"{self.did}.ann"
    with annfile.open('w+') as f:
      for entity in self.entities:
        f.write(entity.to_brat(entity_count))
        f.write('\n')
        entity_count += 1

  def to_json(self):
    return {
      'did': self.did,
      'title': self.title,
      'sentences': [s.to_json() for s in self.sentences]
    }


@dataset_ingredient.capture
def create_dataset(project_dir):
  import json
  from tqdm import tqdm

  datadir = Path(project_dir) / 'data'
  bratdir = datadir / 'brat'
  jsondir = datadir / 'json'
  cui2id = json.load((Path(project_dir) / 'info' / 'cui2id.json').open('r'))

  doc_ids = {}
  for line in (datadir / 'corpus_pubtator_pmids_trng.txt').open('r'):
    doc_ids[line.strip()] = 'train'
  for line in (datadir / 'corpus_pubtator_pmids_dev.txt').open('r'):
    doc_ids[line.strip()] = 'dev'
  for line in (datadir / 'corpus_pubtator_pmids_test.txt').open('r'):
    doc_ids[line.strip()] = 'test'
  doc_counts = {
    'train': 0,
    'dev': 0,
    'test': 0
  }

  doc_lines = []
  total_cuis = 0.
  with_embedding = 0.
  unique_cuis = set()
  with tqdm(total=4392) as pbar:
    for i, line in enumerate((datadir / 'corpus_pubtator.txt').open('r')):
      line = line.strip()
      try:
        if len(line.strip()) > 1:
          doc_lines.append(line.strip())
        else:
          doc = Document(doc_lines)

          # stats
          for concept in doc.entities:
            total_cuis += 1.
            if concept.cui in cui2id:
              with_embedding += 1.
            unique_cuis.add(concept.cui)

          split = doc_ids[doc.did]
          # write brat
          if doc_counts[split] < 50:
            outdir = bratdir / split
            doc.to_brat(outdir)
            doc_counts[split] += 1
            log.info(f"Writing {split} doc {doc.did} {doc_counts[split]}/{50}")
          # write json
          outfile = jsondir / split / f"{doc.did}.json"
          json.dump(doc.to_json(), outfile.open('w+'))
          pbar.update()

          doc_lines = []
      except Exception as e:
        log.error(f"Died on line {i}: |{line}|")
        log.error(e)
        traceback.print_exc()
        exit()

  log.info(f"{with_embedding} of {total_cuis} cui occurrences have an embedding ({with_embedding / total_cuis})")
  unique_with_emb = len([c for c in unique_cuis if c in cui2id])
  log.info(f"{unique_with_emb} of {len(unique_cuis)} unique cuis have an embedding ({float(unique_with_emb) / len(unique_cuis)}")


if __name__ == "__main__":
  main()
