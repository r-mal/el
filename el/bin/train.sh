
python -m el.cli preprocess with \
  dataset.record_dir_name='cake_cls_sep'


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
  estimator.run_name='cake-36' \
  dataset.record_dir_name='cake_cls_sep' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  train.learning_rate=1e-4 \
  dataset.bert_model='ncbi_uncased_base' \
  model.scoring_fn='energy_with_loss' \
  model.string_method='bayesian' \
  model.norm_loss_fn='multinomial_ce_prob' \
  model.type_loss_fn='multinomial_ce' \
  -m bigmem13.hlt.utdallas.edu:27017:el

python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-37' \
  dataset.record_dir_name='cake_cls_sep' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  train.learning_rate=1e-4 \
  dataset.bert_model='pubmed_uncased_base' \
  model.scoring_fn='energy_with_loss' \
  model.string_method='bayesian' \
  model.norm_loss_fn='multinomial_ce_prob' \
  model.type_loss_fn='multinomial_ce' \
  -m bigmem13.hlt.utdallas.edu:27017:el


python -m el.cli preprocess with \
  dataset.record_dir_name='cake' \
  dataset.dataset='clef'
