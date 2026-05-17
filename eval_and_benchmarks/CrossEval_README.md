# Cross-corpus Evaluation — Vienna4x22 & Batik-plays-Mozart

A test-only evaluation of mazurka-trained models on two out-of-domain classical piano corpora. **Built for my PhD thesis** to characterise how well the proposed dynamic / change-point / beat / downbeat predictors generalise beyond the Chopin mazurka training distribution.

Run from [CrossEval.ipynb](CrossEval.ipynb). Numbers feed the cross-corpus tables in the thesis.

---

## 1. Class-label priors

All three corpora share the same 5-level vocabulary (`pp, p, mf, f, ff`); the difference is the **distribution**, which is what we expect to challenge cross-corpus generalisation.

| 5-level | Mazurka (train) | Vienna4x22 | Batik-plays-Mozart |
|---|---:|---:|---:|
| `pp` | 7.0 % | 25.5 % | 0.8 % |
| `p`  | 54.4 % | 44.9 % | 52.9 % |
| `mf` | 5.7 % | 27.0 % | 0.8 % |
| `f`  | 29.9 % | 2.3 % | 45.5 % |
| `ff` | 3.1 % | 0.3 % | 0.1 % |
| # beats | 685 209 | 13 618 | 23 662 |
| # change events | 19 033 | 396 | 2 327 |

- **Vienna** is quiet-skewed (`pp+p+mf` = 97 %); a mazurka model's `f`-heavy prior is wrong here.
- **Batik** is bimodal Mozart (`p`/`f` only).
- **Mazurka** sits in the middle.

## 2. Coarse 3-class F1

Fine-grained 5-class F1 punishes adjacent confusions (`mf` ↔ `f`) the same as far ones (`mf` ↔ `pp`). For cross-corpus reporting we additionally compute a coarse weighted-F1 over **{quiet, medium, loud}**:

| 5-level | coarse |
|---|---|
| `pp`, `p` | quiet  |
| `mf`      | medium |
| `f`, `ff` | loud   |

It's printed alongside the fine F1 as `dynamic_coarse_f1` — no schema change, no separate run.

## 3. How to run

```bash
# one-time: pack the test HDF5s (mirrors mazurka pack_h5 style)
python pytorch/data_preprocess.py --mode pack_h5_vienna --sample_rate 22050
python pytorch/data_preprocess.py --mode pack_h5_batik  --sample_rate 22050
```

Then step through [CrossEval.ipynb](CrossEval.ipynb). Each cell loops 5 mazurka-trained folds against one corpus and prints

```
epoch_*.pth -> {dynamic_f1, dynamic_coarse_f1, change_point_f1, beat_f1, downbeat_f1}
```

per fold. Mean ± Std goes into the tables at the top of the notebook.

Smoke test (fold 0, BSSL + MultiTaskCNN):

| corpus | `dyn_f1` | `dyn_coarse_f1` | `cp_f1` | `beat_f1` | `db_f1` |
|---|---:|---:|---:|---:|---:|
| Vienna4x22 | 0.133 | 0.365 | 0.298 | 0.724 | 0.256 |
| Batik | 0.490 | 0.502 | 0.133 | 0.388 | 0.132 |

The wide Vienna gap (0.13 → 0.37) is the expected "right region, wrong fine class" pattern from the prior mismatch. Batik's fine ≈ coarse because its vocabulary is essentially binary p/f to begin with.

## 4. HDF5 schema

Identical to the mazurka pack. Each `.h5` has `waveform / midi_event* / beat_time / downbeat_time / measure_time / change_point_time / dynmark_beats / dynmark_changes / dynmark_5_class / dynmark_8_class` plus the standard attrs. Vienna and Batik populate them from MusicXML + `.match` via [pytorch/score_features.py](../pytorch/score_features.py) (`partitura` backend).

## 5. Files in this work

| file | role |
|---|---|
| `pytorch/score_features.py` | partitura → mazurka-schema feature extraction |
| `pytorch/data_preprocess.py` | adds `pack_h5_vienna` / `pack_h5_batik` modes |
| `pytorch/config.yaml` | adds `dataset.name`, `dataset.vienna`, `dataset.batik` (with empty `exclude_opus` lists) |
| `pytorch/inference.py` | swap hardcoded `mazurka_sr…` for `{cfg.dataset.name}_sr…` |
| `pytorch/final_evaluation.py` | same swap + adds `dynamic_coarse_f1` |
| `pytorch/utils.py` | `load_model()` uses `OmegaConf.from_dotlist` so `dataset.name=vienna` actually nests |
| `eval_and_benchmarks/CrossEval.ipynb` | per-variant cells, same style as `Eval_Multitask.ipynb` |
| `eval_and_benchmarks/CrossEval_README.md` | this file |

Mazurka in-domain training and `Eval_Multitask.ipynb` / `Eval_Singletask.ipynb` numbers are unaffected.
