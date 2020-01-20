import sys
import numpy as np
import json
from pathlib import Path


def create_type_embeddings():
  project_dir = Path(sys.argv[1])
  emb_name = sys.argv[2]
  do_pool = sys.argv[3] == 'pool'
  cui2id = json.load((project_dir / 'info' / 'cui2id.json').open())
  tui2id = json.load((project_dir / 'info' / 'tui2label.json').open())
  type_id2cui_ids = {int(k): v for k, v in json.load((project_dir / 'info' / 'semtype2cuis.json').open()).items()}
  with np.load(str(project_dir / 'info' / emb_name /'embeddings.npz')) as npz:
    embeddings = npz['embs']

  mean = np.mean(embeddings)
  std = np.std(embeddings)
  type_embeddings = np.random.normal(size=[len(tui2id), embeddings.shape[-1]], loc=mean, scale=std)
  if do_pool:
    for tui, i in tui2id.items():
      tui_id = cui2id[tui]
      if tui_id in type_id2cui_ids:
        cui_ids = type_id2cui_ids[tui_id]  # ids of each cui of this type
        type_embeddings[i] = np.mean(embeddings[cui_ids], axis=0)
        print(f"Averaging {len(cui_ids)} cui embeddings for type {tui}")
      else:
        print(f"TUI {tui} has no associated concepts in semtype2cuis.json!")
  else:
    for tui, i in sorted(tui2id.items()):
      type_embeddings[i] = embeddings[cui2id[tui]]
  print(f"Saving type embedding matrix with shape {type_embeddings.shape}")
  np.savez_compressed(str(project_dir / 'info' / emb_name / 'type_embeddings.npz'),
                      embs=type_embeddings)


if __name__ == "__main__":
  create_type_embeddings()
