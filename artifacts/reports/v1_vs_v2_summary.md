# v1 vs v2 Sleep-Stage Model Comparison

**v1 leader**: `xgboost_cuda` (macroF1=0.6678, kappa=0.5050)
**v2 leader**: `base_catboost_hmm` (macroF1=0.6993, kappa=0.5501)
**Delta**: accuracy: +1.00pp, balanced_accuracy: +5.00pp, macro_f1: +3.15pp, cohen_kappa: +4.52pp

## Full Leaderboard (sorted by macro F1)

| Version | Family | Model | Acc | Bal Acc | Macro F1 | Wtd F1 | Kappa |
|---|---|---|---:|---:|---:|---:|---:|
| v2 | ensemble_v2 | base_catboost_hmm | 0.7746 | 0.7159 | 0.6993 | 0.7793 | 0.5501 |
| v2 | ensemble_v2 | avg_geo_hmm | 0.8035 | 0.6745 | 0.6932 | 0.7978 | 0.5695 |
| v2 | ensemble_v2 | avg_mean_hmm | 0.8022 | 0.6733 | 0.6921 | 0.7965 | 0.5668 |
| v2 | ensemble_v2 | avg_geo | 0.7772 | 0.6973 | 0.6901 | 0.7795 | 0.5388 |
| v2 | ensemble_v2 | avg_mean | 0.7754 | 0.6975 | 0.6892 | 0.7780 | 0.5368 |
| v2 | ensemble_v2 | base_catboost_tuned_hmm | 0.7816 | 0.6870 | 0.6814 | 0.7820 | 0.5515 |
| v2 | ensemble_v2 | base_lightgbm | 0.7796 | 0.6605 | 0.6762 | 0.7761 | 0.5176 |
| v2 | ensemble_v2 | base_catboost_tuned | 0.7538 | 0.7053 | 0.6750 | 0.7615 | 0.5201 |
| v2 | ensemble_v2 | base_lightgbm_hmm | 0.7897 | 0.6332 | 0.6687 | 0.7794 | 0.5153 |
| v1 | ml_3class | xgboost_cuda | 0.7646 | 0.6659 | 0.6678 | 0.7651 | 0.5050 |
| v2 | ensemble_v2 | base_xgboost_cuda_hmm | 0.7766 | 0.6456 | 0.6667 | 0.7715 | 0.5079 |
| v2 | ensemble_v2 | stack_logreg_C0.3 | 0.7902 | 0.6234 | 0.6666 | 0.7762 | 0.4992 |
| v2 | ensemble_v2 | base_xgboost_cuda | 0.7588 | 0.6631 | 0.6646 | 0.7599 | 0.4934 |
| v2 | ensemble_v2 | base_catboost | 0.7369 | 0.7041 | 0.6641 | 0.7470 | 0.4968 |
| v2 | ensemble_v2 | stack_logreg_C0.3_hmm | 0.7973 | 0.6096 | 0.6619 | 0.7795 | 0.5030 |
| v1 | ml_3class | hist_gradient_boosting | 0.7581 | 0.6490 | 0.6583 | 0.7574 | 0.4835 |
| v2 | ensemble_v2 | base_bilstm_attention_v2_hmm | 0.7312 | 0.6460 | 0.6368 | 0.7348 | 0.4462 |
| v2 | ensemble_v2 | base_bilstm_attention_v2 | 0.7201 | 0.6589 | 0.6315 | 0.7273 | 0.4402 |
| v1 | dl_3class | bilstm_attention | 0.7279 | 0.6472 | 0.6305 | 0.7299 | 0.4327 |
| v1 | dl_3class | temporal_cnn | 0.6832 | 0.6694 | 0.6194 | 0.6959 | 0.4160 |
| v1 | dl_3class | tabular_mlp | 0.6877 | 0.6336 | 0.6175 | 0.6970 | 0.3818 |
| v1 | ml_3class | extra_trees | 0.7615 | 0.5440 | 0.5917 | 0.7240 | 0.3695 |
