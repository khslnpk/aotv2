# v1 vs v2 Sleep-Stage Model Comparison

**v1 leader**: `xgboost_cuda` (macroF1=0.6678, kappa=0.5050)
**v2 leader**: `base_catboost_hmm` (macroF1=0.6993, kappa=0.5501)
**Delta**: accuracy: +1.00pp, balanced_accuracy: +5.00pp, macro_f1: +3.15pp, cohen_kappa: +4.52pp

## Full Leaderboard (sorted by macro F1)

| Version | Family | Model | Acc | Bal Acc | Macro F1 | Wtd F1 | Kappa |
|---|---|---|---:|---:|---:|---:|---:|
| v2 | ensemble | base_catboost_hmm | 0.7746 | 0.7159 | 0.6993 | 0.7793 | 0.5501 |
| v2 | ensemble | ensemble_mean | 0.7704 | 0.6783 | 0.6785 | 0.7714 | 0.5178 |
| v2 | ensemble | base_lightgbm | 0.7796 | 0.6605 | 0.6762 | 0.7761 | 0.5176 |
| v2 | ensemble | base_lightgbm_hmm | 0.7897 | 0.6332 | 0.6687 | 0.7794 | 0.5153 |
| v1 | ml_3class | xgboost_cuda | 0.7646 | 0.6659 | 0.6678 | 0.7651 | 0.5050 |
| v2 | ensemble | base_xgboost_cuda_hmm | 0.7766 | 0.6456 | 0.6667 | 0.7715 | 0.5079 |
| v2 | ensemble | base_xgboost_cuda | 0.7588 | 0.6631 | 0.6646 | 0.7599 | 0.4934 |
| v2 | ensemble | base_catboost | 0.7369 | 0.7041 | 0.6641 | 0.7470 | 0.4968 |
| v2 | ensemble | ensemble_stacked_hmm | 0.7150 | 0.7058 | 0.6623 | 0.7278 | 0.4744 |
| v1 | ml_3class | hist_gradient_boosting | 0.7581 | 0.6490 | 0.6583 | 0.7574 | 0.4835 |
| v2 | ensemble | ensemble_stacked | 0.6963 | 0.6936 | 0.6419 | 0.7109 | 0.4461 |
| v1 | dl_3class | bilstm_attention | 0.7279 | 0.6472 | 0.6305 | 0.7299 | 0.4327 |
| v1 | dl_3class | temporal_cnn | 0.6832 | 0.6694 | 0.6194 | 0.6959 | 0.4160 |
| v1 | dl_3class | tabular_mlp | 0.6877 | 0.6336 | 0.6175 | 0.6970 | 0.3818 |
| v1 | ml_3class | extra_trees | 0.7615 | 0.5440 | 0.5917 | 0.7240 | 0.3695 |
