import json
import sys
from pathlib import Path
import numpy as np


def main():
  deffile = sys.argv[1]
  outf = sys.argv[2]

  type2id = {}
  for i, line in enumerate(open(deffile)):
    if i > 126:
      break
    fields = line.split('|')
    type2id[fields[2]] = i

  json.dump(type2id, open(outf, 'w+'))


def main2():
  project_dir = Path(sys.argv[1])
  tui2id = json.load((project_dir / 'info' / 'tui2label.json').open())
  with np.load(str(project_dir / 'info' / 'semtype_text.npz')) as npz:
    tui_embeddings = npz['embs']

  tui_embs2 = np.zeros((len(tui2id), tui_embeddings.shape[-1]), dtype=float)
  for i, emb in enumerate(tui_embeddings):
    tui = list('T000')
    tui[-len(str(i+1)):] = str(i+1)
    tui = ''.join(tui)
    if tui in tui2id:
      tui_embs2[tui2id[tui]] = emb
    else:
      print(f'No id for TUI {tui}!')

  np.savez_compressed((project_dir / 'info' / 'max' / 'type_text_embeddings.npz'), embs=tui_embs2)


if __name__ == "__main__":
  main2()
