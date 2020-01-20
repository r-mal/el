import json
import sys


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


if __name__ == "__main__":
  main()
