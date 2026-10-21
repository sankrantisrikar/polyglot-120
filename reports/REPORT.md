# Polyglot-120 evaluation report

| Slice | N | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| clean | 121440 | 0.9444 | 0.9481 |
| short | 36000 | 0.7758 | 0.7817 |
| noisy | 36000 | 0.7281 | 0.7476 |
| code_mixed | 36000 | 0.7525 | 0.7606 |
| mixed_script | 36000 | 0.7557 | 0.7614 |

## 15 lowest-F1 languages (clean)

| Language | P | R | F1 | n |
|---|---:|---:|---:|---:|
| acq_Arab | 0.000 | 0.000 | 0.000 | 1012 |
| apc_Arab | 0.407 | 0.143 | 0.212 | 1012 |
| acm_Arab | 0.249 | 0.271 | 0.259 | 1012 |
| yue_Hant | 0.688 | 0.292 | 0.411 | 1012 |
| aeb_Arab | 0.601 | 0.376 | 0.463 | 1012 |
| arb_Arab | 0.341 | 0.947 | 0.501 | 1012 |
| bos_Latn | 0.672 | 0.453 | 0.541 | 1012 |
| hrv_Latn | 0.587 | 0.776 | 0.668 | 1012 |
| zho_Hant | 0.552 | 0.867 | 0.674 | 1012 |
| arz_Arab | 0.614 | 0.770 | 0.683 | 1012 |
| ary_Arab | 0.858 | 0.770 | 0.811 | 1012 |
| tzm_Tfng | 0.955 | 0.851 | 0.900 | 1012 |
| sat_Olck | 0.861 | 0.968 | 0.912 | 1012 |
| ind_Latn | 0.925 | 0.913 | 0.919 | 1012 |
| zsm_Latn | 0.914 | 0.927 | 0.921 | 1012 |

## Top confusion pairs (clean)

| True | Predicted | Count |
|---|---|---:|
| acq_Arab | arb_Arab | 718 |
| yue_Hant | zho_Hant | 713 |
| acm_Arab | arb_Arab | 628 |
| bos_Latn | hrv_Latn | 551 |
| apc_Arab | acm_Arab | 309 |
| apc_Arab | arb_Arab | 265 |
| hrv_Latn | bos_Latn | 224 |
| apc_Arab | arz_Arab | 216 |
| acq_Arab | acm_Arab | 194 |
| aeb_Arab | acm_Arab | 185 |
| aeb_Arab | arb_Arab | 155 |
| tzm_Tfng | sat_Olck | 151 |
| aeb_Arab | arz_Arab | 145 |
| zho_Hant | yue_Hant | 134 |
| bod_Tibt | dzo_Tibt | 121 |
| aeb_Arab | ary_Arab | 99 |
| ary_Arab | aeb_Arab | 88 |
| ind_Latn | zsm_Latn | 88 |
| zsm_Latn | ind_Latn | 74 |
| arz_Arab | acm_Arab | 71 |

## Confidence routing (clean): pick your fallback threshold

| Threshold | Coverage | Acc on kept | Acc on routed |
|---:|---:|---:|---:|
| 0.5 | 0.944 | 0.9814 | 0.3871 |
| 0.6 | 0.916 | 0.9925 | 0.4633 |
| 0.7 | 0.909 | 0.9939 | 0.4900 |
| 0.8 | 0.905 | 0.9950 | 0.5008 |
| 0.9 | 0.891 | 0.9976 | 0.5447 |
| 0.95 | 0.853 | 0.9989 | 0.6541 |
