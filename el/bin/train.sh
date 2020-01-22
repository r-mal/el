
python -m el.cli preprocess with \
  dataset.record_dir_name='cake'

python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-1' \
  dataset.record_dir_name='cake' \
  dataset.tasks="[cake]" \
  -m bigmem13.hlt.utdallas.edu:27017:el

# TODO
python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-2' \
  dataset.record_dir_name='cake' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  dataset.bert_model='base_uncased' \
  -m bigmem13.hlt.utdallas.edu:27017:el

python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-3' \
  dataset.record_dir_name='cake' \
  dataset.tasks="[cake,type]" \
  dataset.batch_size=12 \
  dataset.bert_model='base_uncased' \
  -m bigmem13.hlt.utdallas.edu:27017:el

python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-4' \
  dataset.record_dir_name='cake' \
  dataset.tasks="[cake,type]" \
  dataset.batch_size=12 \
  dataset.bert_model='ncbi_uncased_base' \
  -m bigmem13.hlt.utdallas.edu:27017:el

python -m el.cli preprocess with \
  dataset.record_dir_name='cake' \
  dataset.dataset='clef'
