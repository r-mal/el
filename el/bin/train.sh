
python -m el.cli preprocess with \
  dataset.record_dir_name='cake'

python -m el.cli train with \
  dataset.record_dir_name='cake' \
  dataset.tasks="[cake]" \
  -m bigmem13.hlt.utdallas.edu:27017:el



python -m el.cli preprocess with \
  dataset.record_dir_name='cake' \
  dataset.dataset='clef'
