
python -m el.cli preprocess with \
  dataset.record_dir_name='cake'


python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-32' \
  dataset.record_dir_name='cake' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  dataset.bert_model='ncbi_uncased_base' \
  model.scoring_fn='energy_with_loss' \
  model.string_method='bayesian' \
  model.norm_loss_fn='multinomial_ce_prob' \
  model.type_loss_fn='multinomial_ce' \
  -m bigmem13.hlt.utdallas.edu:27017:el

python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-33' \
  dataset.record_dir_name='cake' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  dataset.bert_model='ncbi_uncased_base' \
  model.scoring_fn='energy_with_loss' \
  model.string_method='weighted_scores' \
  model.norm_loss_fn='multinomial_ce' \
  model.type_loss_fn='multinomial_ce' \
  -m bigmem13.hlt.utdallas.edu:27017:el


python -m el.cli preprocess with \
  dataset.record_dir_name='cake' \
  dataset.dataset='clef'
