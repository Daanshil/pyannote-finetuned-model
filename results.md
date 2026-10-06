# Experimental Results and Evaluation

## 1. Does adaptation generalise across datasets?

All adapted models are trained with a 10 s segment length (the native chunk length of the pretrained segmentation model). Evaluated on the Babaloon and UP-child development sets.

| Training data | DER Babaloon | DER UP-child | JER Babaloon | JER UP-child |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | 14.76 | 24.43 | 18.04 | 29.88 |
| **UP-child** | 16.35 | **13.15** | 20.20 | 13.08 |
| **Babaloon** | **5.85** | 21.53 | **7.39** | 25.68 |
| **Combined** | 6.76 | **12.58** | 9.40 | **12.52** |


---

## 2. Impact of training segment duration


| Duration | DER Babaloon | DER UP-child | JER Babaloon | JER UP-child |
| :--- | :---: | :---: | :---: | :---: |
| **5 s** | 8.82 | 25.13 | 13.13 | 33.17 |
| **10 s** | 6.76 | 12.58 | 9.40 | 12.52 |
| **15 s** | 6.61 | 13.04 | 8.07 | 12.98 |
| **20 s** | **6.23** | **12.16** | **7.55** | 12.38 |
| **30 s** | 6.32 | 12.57 | 7.73 | 12.86 |
| **40 s** | 6.76 | 12.77 | 8.12 | 12.65 |
| **50 s** | 6.52 | 12.58 | 7.90 | **12.35** |
| **60 s** | 6.65 | 13.67 | 8.61 | 14.37 |

Error drops sharply from 5 s to 10 s (especially on UP-child, where DER halves from 25.13% to 12.58%) and is essentially flat beyond 10 s. The best overall trade-off is around 15–20 s.

---

## 3. Robustness of clustering and post-processing hyperparameters

Combined segmentation model fixed; each parameter varied in turn with the others held at their best values. Evaluated on the development sets (Babaloon / UP-child).

### 3.1 Minimum duration off (s)

| Min duration off | DER Babaloon | DER UP | JER Babaloon | JER UP | Purity Babaloon | Purity UP | Coverage Babaloon | Coverage UP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0** | 6.61 | 13.04 | 8.07 | 12.98 | 93.92 | 93.20 | 94.64 | 88.57 |
| **0.1** | 6.58 | 13.03 | 8.05 | 12.95 | 93.86 | 93.16 | 94.70 | 88.61 |
| **0.2** | 6.38 | 12.93 | 7.83 | 12.81 | 93.61 | 93.06 | 94.97 | 88.77 |
| **0.3** | **6.33** | 12.56 | **7.79** | 12.44 | 92.90 | 92.80 | 95.30 | 89.24 |
| **0.4** | 6.64 | 12.28 | 8.10 | 12.14 | 91.83 | 92.29 | 95.61 | 89.81 |
| **0.5** | 7.32 | **12.11** | 8.72 | 11.95 | 90.74 | 91.71 | 95.75 | 90.39 |
| **0.6** | 8.43 | **12.11** | 9.61 | **11.90** | 89.36 | 91.14 | 95.89 | 90.84 |
| **0.7** | 9.48 | 12.17 | 10.44 | 12.00 | 88.16 | 90.36 | 96.02 | 91.38 |
| **0.8** | 10.64 | 12.58 | 11.33 | 12.36 | 86.96 | 89.52 | 96.12 | 91.72 |
| **0.9** | 11.73 | 13.04 | 12.09 | 12.81 | 85.89 | 88.57 | 96.25 | 92.13 |
| **1.0** | 12.84 | 14.00 | 12.93 | 13.57 | 84.84 | 87.49 | 96.34 | 92.19 |

Babaloon DER is minimised around 0.3 s; UP-child DER keeps improving to roughly 0.5–0.6 s before degrading. Raising the threshold lowers purity (9.1 points on Babaloon, 5.7 on UP-child) while raising coverage (1.7 and 3.6 points).

### 3.2 Linkage method

| Linkage | DER Babaloon | DER UP | JER Babaloon | JER UP | Purity Babaloon | Purity UP | Coverage Babaloon | Coverage UP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centroid** | 6.60 | **13.05** | 7.93 | **13.15** | 93.22 | 92.83 | 94.43 | 88.49 |
| **Average** | 6.60 | 16.23 | 7.93 | 17.83 | 93.22 | 89.86 | 94.43 | 88.16 |
| **Single** | 42.47 | 44.79 | 70.24 | 68.87 | 59.62 | 60.67 | 95.70 | 93.53 |

Centroid is best; single linkage is worst. Its high coverage reflects near-total cluster collapse rather than genuine coverage.

### 3.3 Agglomerative clustering distance threshold

| Threshold | DER Babaloon | DER UP | JER Babaloon | JER UP | Purity Babaloon | Purity UP | Coverage Babaloon | Coverage UP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.3** | 42.47 | 44.80 | 70.24 | 68.87 | 59.62 | 60.67 | 95.69 | 93.53 |
| **0.4** | 36.34 | 44.80 | 58.94 | 68.87 | 66.36 | 60.67 | 92.30 | 93.53 |
| **0.5** | 24.18 | 44.80 | 38.14 | 68.87 | 80.00 | 60.67 | 92.39 | 93.53 |
| **0.6** | 6.60 | 23.53 | 8.07 | 31.13 | 93.23 | 82.61 | 94.64 | 90.32 |
| **0.7** | 6.60 | 13.60 | 8.07 | 13.58 | 93.23 | 93.20 | 94.64 | 87.97 |
| **0.8** | 6.60 | **13.13** | **7.93** | **12.58** | 93.23 | **93.28** | 94.64 | 88.41 |
| **0.9** | 6.60 | 13.60 | 7.93 | 13.58 | 93.23 | 93.20 | 94.64 | 87.97 |

Below 0.6, nearly all speakers collapse into a single cluster. Values from 0.7 to 0.9 form a stable plateau, with 0.8 giving the best UP-child result.

**Chosen configuration for test experiments:** threshold 0.8, minimum cluster size 20, centroid linkage.

---

## 4. Held-out test results (Babaloon test set)

Best configuration above. Fine-tuned rows use the combined training set at a 15 s segment length.

| Configuration | DER (%) | JER (%) |
| :--- | :---: | :---: |
| **Baseline, no collar** | 26.11 | 29.25 |
| **Baseline, with collar** | 17.75 | 22.91 |
| **15 s, no collar** | 11.18 | 14.58 |
| **15 s, with collar** | **6.16** | **9.67** |
| **15 s, with collar and min duration off = 0.3** | 6.33 | 9.81 |

Fine-tuning reduces DER by roughly 60% without a collar (26.11 → 11.18). Applying the 0.25 s collar lowers DER by about 4.8 points on the fine-tuned model. The 0.3 s minimum-duration-off step makes little difference (at most 0.17 points).