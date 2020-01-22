from typing import List
from pathlib import Path
import traceback
import json
from tqdm import tqdm
from hedgedog.logging import get_logger
from hedgedog.nlp.spacy.umls import UmlsCandidateGenerator
from hedgedog.tf.estimator.ingredients import dataset_ingredient
from el.data.text import Concept, Document, Span

log = get_logger("mm.data.medmentions")


class MedMentionsDocument(Document):
  def __init__(self, lines: List[str], umls, k, mention2idx):
    did, _, title = lines[0].strip().split('|')
    text = '|'.join(lines[1].strip().split('|')[2:])
    doc_string = title + '. ' + text
    entities = []
    for eid, line in enumerate(lines[2:]):
      did, start, end, text, types, cui = line.strip().split('\t')
      cui = cui.replace('UMLS:', '')
      start, end = int(start), int(end)
      if start > len(title):
        start += 1
        end += 1
      entities.append(Concept([Span(start, end, text)], types.split(','), cui))

    super().__init__(did, doc_string, entities, umls, k, mention2idx)


@dataset_ingredient.capture
def create_dataset(project_dir, data_dir, candidates_per_concept):
  datadir = Path(project_dir) / 'medmentions'
  bratdir = datadir / 'brat'
  jsondir = Path(data_dir)
  cui2id = json.load((Path(project_dir) / 'info' / 'cui2id.json').open('r'))
  mention2idx = json.load((Path(project_dir) / 'info' / 'medmentions_mentions.json').open('r'))

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
  umls = UmlsCandidateGenerator()
  with tqdm(total=4392) as pbar:
    for i, line in enumerate((datadir / 'corpus_pubtator.txt').open('r')):
      line = line.strip()
      try:
        if len(line.strip()) > 1:
          doc_lines.append(line.strip())
        else:
          doc = MedMentionsDocument(doc_lines, umls, candidates_per_concept, mention2idx)

          # stats
          for concept in doc.concepts:
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
  log.info(f"{unique_with_emb} of {len(unique_cuis)} unique cuis "
           f"have an embedding ({float(unique_with_emb) / len(unique_cuis)}")


@dataset_ingredient.capture
def dump_mentions(project_dir):
  datadir = Path(project_dir) / 'medmentions'
  mentions = set()

  for line in (datadir / 'corpus_pubtator.txt').open('r'):
    line = line.strip()
    if len(line) > 1:
      fields = line.split('\t')
      if len(fields) == 6:
        mentions.add(fields[3].strip())
  out_file = Path(project_dir) / 'info' / 'medmentions_mentions.json'
  log.info(f"Writing {len(mentions)} mentions to {out_file}")
  json.dump({m: i for i, m in enumerate(mentions)}, out_file.open('w+'))
