
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
  estimator.run_name='cake-40' \
  dataset.record_dir_name='cake_cls_sep' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  train.learning_rate=5e-5 \
  dataset.bert_model='ncbi_uncased_base' \
  model.scoring_fn='energy' \
  model.margin=0.1 \
  model.use_string_sim=False \
  model.norm_loss_fn='margin_loss' \
  -m bigmem13.hlt.utdallas.edu:27017:el

python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-42' \
  dataset.record_dir_name='cake_cls_sep' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  train.learning_rate=5e-5 \
  train.gradient_clip=1.0 \
  dataset.bert_model='base_uncased' \
  model.scoring_fn='energy' \
  model.margin=0.2 \
  model.use_string_sim=False \
  model.norm_loss_fn='margin_loss' \
  model.train_bert=False \
  -m bigmem13.hlt.utdallas.edu:27017:el



python -m el.cli preprocess with \
  seed=1337 \
  dataset.record_dir_name='cake_cls_sep_mention_candidates' \
  dataset.mention_candidate_path='/users/rmm120030/working/kge_ner/info/knn-emb.npz'

python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-39' \
  dataset.record_dir_name='cake_cls_sep_mention_candidates' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  train.learning_rate=1e-4 \
  train.gradient_clip=1.0 \
  dataset.bert_model='ncbi_uncased_base' \
  model.scoring_fn='energy' \
  model.string_method='bayesian' \
  model.norm_loss_fn='multinomial_ce_prob' \
  model.type_loss_fn='multinomial_ce' \
  dataset.mention_candidate_path='/users/rmm120030/working/kge_ner/info/knn-emb.npz' \
  -m bigmem13.hlt.utdallas.edu:27017:el


python -m el.cli preprocess with \
  dataset.record_dir_name='cake' \
  dataset.dataset='clef'
