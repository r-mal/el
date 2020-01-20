from pathlib import Path
from hedgedog.logging import get_logger
from hedgedog.nlp.spacy.umls import UmlsCandidateGenerator
from hedgedog.tf.estimator.ingredients import dataset_ingredient
from mm.data.text import Concept, Document, Span
import json
from tqdm import tqdm
import traceback

log = get_logger("mm.data.clef")


class ClefDocument(Document):
  def __init__(self, data_dir: Path, doc_id: str, umls, k, code2id, id2code, id2types, mention2idx):
    text = (data_dir / f"{doc_id}.text").read_text().replace('\t', ' ')
    concepts = []
    with (data_dir / f"{doc_id}.pipe").open('r') as f:
      for line in f:
        line = line.strip()
        try:
          fields = line.split('||')
          _, cui = fields[1], fields[2]
          spans = []
          for i in range(3, len(fields), 2):
            start, end = int(fields[i]), int(fields[i + 1])
            spans.append(Span(start, end, text[start:end]))
        except ValueError as e:
          log.error(f"Could not parse line: ##{line}##")
          log.error(e)
          traceback.print_exc()
          exit()

        # grab semantic types
        types = ['UnknownType']
        if cui in code2id and code2id[cui] in id2types:
          types = [id2code[t] for t in id2types[code2id[cui]]]
        concepts.append(Concept(spans, types, cui))
    super().__init__(doc_id, text, concepts, umls, k, mention2idx)


@dataset_ingredient.capture
def create_dataset(project_dir, data_dir, candidates_per_concept):
  raw_dir = Path(project_dir) / 'clef'
  dataset_dir = Path(data_dir)
  dev_ids = [x.strip() for x in (raw_dir / 'dev_files.txt').open()]
  train_ids = [x.replace('train/', '').replace('.text', '').strip() for x in (raw_dir / 'train_files.txt').open('r')
               if x.replace('train/', '').replace('.text', '').strip() not in dev_ids]
  test_ids = [x.replace('test/', '').replace('.text', '').strip() for x in (raw_dir / 'test_files.txt').open('r')]

  umls = UmlsCandidateGenerator()
  cui2id = json.load((Path(project_dir) / 'info' / 'cui2id.json').open())
  mention2idx = json.load((Path(project_dir) / 'info' / 'clef_mentions.json').open())
  id2types = {int(k): v for k, v in json.load((Path(project_dir) / 'info' / 'cui2semtypes.json').open()).items()}
  id2cui = {v: k for k, v in cui2id.items()}

  def process_documents(doc_ids, in_dir, brat_dir, json_dir):
    num = 0
    for did in tqdm(doc_ids, total=len(doc_ids)):
      doc = ClefDocument(in_dir, did, umls, candidates_per_concept, cui2id, id2cui, id2types, mention2idx)
      if num < 20:
        doc.to_brat(brat_dir)
        num += 1
      outfile = json_dir / f"{did}.json"
      json.dump(doc.to_json(), outfile.open('w+'))

  log.info("Processing Training set...")
  process_documents(train_ids, raw_dir / 'train', raw_dir / 'brat' / 'train', dataset_dir / 'train')
  log.info("Processing Dev set...")
  process_documents(dev_ids, raw_dir / 'train', raw_dir / 'brat' / 'dev', dataset_dir / 'dev')
  log.info("Processing Test set...")
  process_documents(test_ids, raw_dir / 'test', raw_dir / 'brat' / 'test', dataset_dir / 'test')


@dataset_ingredient.capture
def dump_mentions(project_dir):
  raw_dir = Path(project_dir) / 'clef'
  train_ids = [x.replace('train/', '').replace('.text', '').strip() for x in (raw_dir / 'train_files.txt').open('r')]
  test_ids = [x.replace('test/', '').replace('.text', '').strip() for x in (raw_dir / 'test_files.txt').open('r')]

  mentions = set()

  def process_documents(doc_ids, in_dir):
    for doc_id in tqdm(doc_ids, total=len(doc_ids)):
      text = (in_dir / f"{doc_id}.text").read_text().replace('\t', ' ')
      with (in_dir / f"{doc_id}.pipe").open('r') as f:
        for line in f:
          line = line.strip()
          try:
            fields = line.split('||')
            spans = []
            for i in range(3, len(fields), 2):
              start, end = int(fields[i]), int(fields[i + 1])
              spans.append(text[start:end].strip())
          except ValueError as e:
            log.error(f"Could not parse line: ##{line}##")
            log.error(e)
            traceback.print_exc()
            exit()
          mentions.add(' '.join(spans))

  process_documents(train_ids, raw_dir / 'train')
  process_documents(test_ids, raw_dir / 'test')

  out_file = Path(project_dir) / 'info' / 'clef_mentions.json'
  json.dump({m: i for i, m in enumerate(mentions)}, out_file.open('w+'))
