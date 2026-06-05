# Experimental Results and Evaluation

This document presents a comprehensive evaluation of various speaker diarisation models and fine-tuning strategies. Performance is assessed using two standard metrics: **Diarisation Error Rate (DER)** and **Jaccard Error Rate (JER)**. Lower values indicate superior performance for both metrics.

The evaluation covers baseline performance, dataset-specific fine-tuning (`babaloon` and `up_child`), joint training configurations (`up+babaloon`), and an ablation study investigating the impact of training chunk segment durations.

---

## 1. Core Diarisation Performance Metrics

### 1.1 Diarisation Error Rate (DER)
The table below details the DER (%) across the development and test subsets for the baseline model (`speaker-diarisation 3.1`), dataset-specific fine-tuned models (trained with 5-second and 15-second chunks), and joint multi-domain fine-tuned models.

| Model | babaloon_dev | babaloon_test | up_child_dev<br>(5 mins) | up_child_test |
| :--- | :---: | :---: | :---: | :---: |
| **speaker-diarisation 3.1** *(Baseline)* | 21.83 | 26.11 | 28.82 | 24.09 |
| **fine-tuned babaloon 5s** | 12.81 | 11.28 | 33.99 | 18.41 |
| **finetune babaloon_15s** | 9.81 | 11.98 | 25.46 | 17.46 |
| **fine-tuned up_child 5s** | 19.54 | 16.07 | 28.30 | 12.86 |
| **finetune up_child 15s** | 17.30 | 16.72 | 16.08 | 9.26 |
| **up+babaloon finetune 5s** | 13.00 | 11.18 | 28.36 | 12.06 |
| **up+babaloon finetune 15s** | **10.80** | **12.57** | **16.32** | **9.40** |

### 1.2 Jaccard Error Rate (JER)
The table below tracks the Jaccard Error Rate (%) under identical evaluation settings to account for speaker-per-frame classification accuracy balances.

| Model | babaloon_dev | babaloon_test | up_child_dev<br>(5 mins) | up_child_test |
| :--- | :---: | :---: | :---: | :---: |
| **speaker-diarisation 3.1** *(Baseline)* | 21.83 | 26.11 | 33.44 | 24.09 |
| **fine-tuned babaloon 5s** | 16.60 | 14.70 | 41.57 | 22.61 |
| **finetune babaloon_15s** | 11.28 | 17.01 | 29.25 | 25.12 |
| **fine-tuned up_child 5s** | 24.74 | 20.42 | 35.21 | 14.50 |
| **finetune up_child 15s** | 18.67 | 21.63 | 15.63 | 11.87 |
| **up+babaloon finetune 5s** | 17.16 | 14.65 | 35.69 | 14.23 |
| **up+babaloon finetune 15s** | **12.19** | **17.77** | **16.15** | **12.08** |


---

## 2. Impact of Training Segment Duration (Ablation Study)


| Experiment (Duration) | Test Set DER (%) | Test Set JER (%) | Development Set DER (%) | Development Set JER (%) |
| :--- | :---: | :---: | :---: | :---: |
| **5s** | 13.06 | 14.62 | 28.49 | 35.87 |
| **10s** | 9.89 | 12.52 | 16.35 | 16.13 |
| **15s** | **9.51** | **12.03** | 16.38 | **16.07** |
| **30s** | 9.61 | 12.17 | **16.01** | 16.09 |
| **40s** | 10.20 | 13.17 | 16.81 | 17.23 |
| **60s** | 9.94 | 12.78 | 16.40 | 16.95 |

