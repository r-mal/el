
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
  estimator.run_name='cake-52' \
  dataset.record_dir_name='cake_cls_sep_c50' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  train.learning_rate=5e-5 \
  train.gradient_clip=1.0 \
  dataset.bert_model='ncbi_uncased_base' \
  model.scoring_fn='dot' \
  model.use_string_sim=True \
  model.string_method='bayesian_norm' \
  model.norm_loss_fn='multinomial_ce_prob' \
  model.train_bert=True \
  -m bigmem13.hlt.utdallas.edu:27017:el

python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-53' \
  dataset.record_dir_name='cake_cls_sep_c53' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  train.learning_rate=5e-5 \
  train.gradient_clip=1.0 \
  dataset.bert_model='ncbi_uncased_base' \
  model.scoring_fn='energy_norm' \
  model.use_string_sim=True \
  model.string_method='exp_weighted' \
  model.norm_loss_fn='multinomial_ce' \
  model.train_bert=True \
  -m bigmem13.hlt.utdallas.edu:27017:el


python -m el.cli preprocess with \
  seed=1337 \
  dataset.record_dir_name='cake_cls_sep_mention_candidates' \
  dataset.mention_candidate_path='/users/rmm120030/working/kge_ner/info/knn-emb.npz'

python -m el.cli preprocess with \
  seed=1337 \
  dataset.record_dir_name='cake_cls_sep_c50' \
  dataset.candidates_per_concept=50

python -m el.cli preprocess with \
  seed=1337 \
  dataset.record_dir_name='cake_cls_sep_c20' \
  dataset.candidates_per_concept=20


python -m el.cli train with \
  seed=1337 \
  estimator.run_name='cake-47' \
  dataset.record_dir_name='cake_cls_sep_mention_candidates' \
  dataset.tasks="[cake]" \
  dataset.batch_size=12 \
  train.learning_rate=5e-5 \
  train.gradient_clip=1.0 \
  dataset.bert_model='ncbi_uncased_base' \
  model.scoring_fn='energy' \
  model.use_string_sim=True \
  model.string_method='bayesian' \
  model.norm_loss_fn='multinomial_ce_prob' \
  model.type_loss_fn='multinomial_ce' \
  dataset.mention_candidate_path='/users/rmm120030/working/kge_ner/info/knn-emb.npz' \
  -m bigmem13.hlt.utdallas.edu:27017:el


python -m el.cli preprocess with \
  dataset.record_dir_name='clef' \
  dataset.dataset='clef'
