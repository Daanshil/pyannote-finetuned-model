# Pyannote Fine-Tuned Model

This repository is dedicated to fine-tuning the `segmentation-3.0` model from Pyannote specifically for Afrikaans child speech.

## Table of Contents

- [Pyannote Fine-Tuned Model](#pyannote-fine-tuned-model)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Setup](#setup)
  - [Config Files](#config-files)
    - [Manual Configuration Template](#manual-configuration-template)
  - [Usage](#usage)
    - [Environment Setup](#environment-setup)
  - [Parameter Guide](#parameter-guide)
    - [General Parameters](#general-parameters)
    - [Training-Specific Parameters](#training-specific-parameters)
    - [Baseline Scoring](#baseline-scoring)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Multi Setup](#multi-setup)
    - [1. Multi Environment Verification](#1-multi-environment-verification)
    - [2. Multi Baseline Scoring](#2-multi-baseline-scoring)
    - [3. Multi Training](#3-multi-training)
    - [4. Multi Fine-Tuned Evaluation](#4-multi-fine-tuned-evaluation)
  - [Automation via Run Scripts](#automation-via-run-scripts)
    - [Execution and Telemetry Logging](#execution-and-telemetry-logging)
    - [Telemetry Pipeline Output](#telemetry-pipeline-output)
  - [Project Structure](#project-structure)
  - [Notes](#notes)

---

## Overview

This repository contains scripts and configuration files to train a fine-tuned Pyannote diarisation model and score diarisation performance.

## Setup

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .finetune_venv
source .finetune_venv/bin/activate
pip install -r requirements.txt

```

---

## Config Files

This implementation requires a `database.yml` file, which is used by `pyannote.database`. A `database.yml` file outlines the absolute file system paths of your training, testing, and development annotations alongside their corresponding audio files. It also requires list files (`.lst`) containing the URI of each audio file.

You can use the helper script inside the `utils/` folder to generate this file automatically:

```bash
python utils/database.py \
  --wav_dir /absolute/path/to/wavs \
  --protocol_name my_protocol \
  --train_lst /absolute/path/to/train.lst \
  --train_rttm "/absolute/path/to/rttm/{uri}.rttm" \
  --dev_lst /absolute/path/to/dev.lst \
  --dev_rttm "/absolute/path/to/rttm/dev_{uri}.rttm" \
  --test_lst /absolute/path/to/test.lst \
  --test_rttm "/absolute/path/to/rttm/{uri}.rttm" \
  --output custom_database.yml

```

### Manual Configuration Template

Alternatively, you can manually construct or update the file using the following template structure (**Note:** absolute paths are required):

```yaml
Databases:
  <database_name>:
    - <wav_dir>/{uri}.wav

Protocols:
  <database_name>:
    SpeakerDiarization:
      <protocol_name>:
        train:
          uri: <train_lst_path>
          annotation: <train_rttm_path>/{uri}.rttm

        development:
          uri: <dev_lst_path>
          annotation: <dev_rttm_path>/{uri}.rttm

        test:
          uri: <test_lst_path>
          annotation: <test_rttm_path>/{uri}.rttm

```

---

## Usage

### Environment Setup

To verify your environment setup and validate your configuration before running workloads, run:

```bash
python setup/setup.py \
  --config /absolute/path/to/database.yml \
  --wav-dir /absolute/path/to/wav/directory \
  --protocol-name Babaloon.SpeakerDiarization.CustomProtocol

```

This script checks if your annotation and audio files are detected successfully. Once validated, specific dataset splits can be safely targeted programmatically via:

* `protocol_name.train`
* `protocol_name.test`
* `protocol_name.development`

---

## Parameter Guide

This guide details the most common command-line parameters used across scripts. All configurations standardise on British English spelling.

### General Parameters

* `--config`: Path to the `database.yml` file defining your datasets (absolute paths recommended).


* `--wav-dir`: Optional path to the directory containing audio files if they are not explicitly hardcoded in the database config.


* `--token`: Your Hugging Face access token (required to download pretrained models and pipelines).


* `--exp-label`: A short identifier string for your experiment — used for naming logs, checkpoints, and output folders.


* `--protocol-name`: The exact name of the Pyannote protocol to use (e.g., `Babaloon.SpeakerDiarization.CustomProtocol`).


* `--protocols`: A space-separated list of multiple protocols used for combination training or scoring.


* `--set`: Which dataset split to target; must be one of `train`, `development`, or `test` (default: `test`).


* `--use-cuda`: Include this flag to run execution on GPU via CUDA if available.


* `--output-dir`: Root output directory where execution results will be saved (default: `output`).



### Training-Specific Parameters

* `--duration`: Length in seconds of training chunks sampled from each audio file (default: `15.0`).


* `--batch-size`: Number of audio chunks processed per training batch (default: `16`).


* `--lr`: Optimiser learning rate (default: `1e-4`).


* `--log-dir`: Directory path where telemetry logs and model checkpoints are saved (default: `logs_15s`).


* `--patience`: Early stopping patience measured in epochs (default: `10`).


* `--epochs`: Maximum number of training epochs to execute (default: `20`).


* `--num-workers`: Number of parallel worker processes to utilise for data loading (default: `2`).



---

### Baseline Scoring

Baseline scoring runs the pretrained **Pyannote Speaker Diarisation 3.1** pipeline to establish a performance benchmark.

> [!IMPORTANT]
> This requires a valid Hugging Face token and user acceptance of the model terms on the Hugging Face repository hub page.
> 
> 

Run the baseline evaluation to generate standard RTTM predictions and performance metrics:

```bash
python scoring/score_default.py \
  --config /absolute/path/to/database.yml \
  --token hf_your_token_here \
  --exp-label baseline_benchmark \
  --set test \
  --output-dir /absolute/path/to/output \
  --protocol-name Babaloon.SpeakerDiarization.CustomProtocol \
  --use-cuda

```

### Training

To train a new fine-tuned segmentation model, update the paths and values below to match your hardware and dataset specifications:

```bash
python training/train.py \
  --config /absolute/path/to/database.yml \
  --wav-dir /absolute/path/to/wavs \
  --token hf_your_token_here \
  --exp-label single_protocol_run \
  --protocol-name protocol.SpeakerDiarization.CustomProtocol \
  --duration 5.0 \
  --batch-size 32 \
  --lr 1e-4 \
  --log-dir /absolute/path/to/logs/up_model_test \
  --patience 10 \
  --num-workers 4 \
  --epochs 20 \
  --use-cuda

```

### Evaluation

Evaluate the performance of your fine-tuned model. This script swaps out the default segmentation component inside the standard pipeline with your local custom-trained checkpoint:

```bash
python scoring/score_model.py \
  --config /absolute/path/to/database.yml \
  --wav-dir /absolute/path/to/wavs \
  --protocol-name Babaloon.SpeakerDiarization.CustomProtocol \
  --token hf_your_token_here \
  --model-path /absolute/path/to/logs/up_model_test/checkpoint.ckpt \
  --exp-label finetuned_evaluation \
  --set test \
  --use-cuda

```
Note: Cross Evaluation

To evalate a model of one protocol onto anotherm and its correcponfing test or dev set. Simply change the protocol name, set and config. It will run the model on that protocol.  

---

## Multi Setup

If you want to fuse multiple dataset protocols together and process them jointly, use the `multi` configuration variants. This architecture relies on a composite `CombinedProtocol` generator layer to safely map mixed paths.

### 1. Multi Environment Verification

Ensure your dataset components are loaded and correctly verified across the fused schema:

```bash
python setup/multi_setup.py \
  --config /absolute/path/to/multi.yml \
  --protocols Babaloon.SpeakerDiarization.CustomProtocol up_child.SpeakerDiarization.CustomProtocol

```

### 2. Multi Baseline Scoring

Evaluate performance benchmarks simultaneously across the combined pipeline:

```bash
python scoring/score_default_multi.py \
  --config /absolute/path/to/multi.yml \
  --token hf_your_token_here \
  --exp-label multi_baseline_benchmark \
  --set test \
  --output-dir /absolute/path/to/output \
  --protocols Babaloon.SpeakerDiarization.CustomProtocol up_child.SpeakerDiarization.CustomProtocol \
  --use-cuda

```

### 3. Multi Training

Fine-tune the central Pyannote segmentation architecture using multi-protocol datasets:

```bash
python training/multi_train.py \
  --config /absolute/path/to/multi.yml \
  --token hf_your_token_here \
  --exp-label multi_protocol_run \
  --duration 15.0 \
  --batch-size 16 \
  --lr 1e-4 \
  --log-dir /absolute/path/to/logs_multi \
  --patience 10 \
  --epochs 20 \
  --num-workers 4 \
  --protocols Babaloon.SpeakerDiarization.CustomProtocol up_child.SpeakerDiarization.CustomProtocol \
  --use-cuda

```

### 4. Multi Fine-Tuned Evaluation

Score your custom checkpoint across the combined datasets:

```bash
python scoring/score_model_multi.py \
  --config /absolute/path/to/multi.yml \
  --token hf_your_token_here \
  --model-path /absolute/path/to/logs_multi/checkpoint.ckpt \
  --exp-label multi_finetuned_eval \
  --set test \
  --output-dir /absolute/path/to/output \
  --protocols Babaloon.SpeakerDiarization.CustomProtocol up_child.SpeakerDiarization.CustomProtocol \
  --use-cuda

```

---

## Automation via Run Scripts

For seamless end-to-end processing execution, shell automation scripts are provided in the repository root directory.

* **`run.sh`**: Targets single-protocol development pipelines.


* **`run_multi.sh`**: Targets multi-protocol combined workflows.



Both automation environments parse input configurations and move dynamically through the following step phases sequentially:


### Execution and Telemetry Logging

To run an automated workflow, configure the internal variables at the head of the respective file and invoke it directly from your terminal terminal:

```bash
bash run.sh
```

### Telemetry Pipeline Output

Every run creates an export runtime folder containing logs and metrics. A central pipeline log dashboard file is compiled directly during execution:

* Single setup tracking outputs save to: `output/{EXP_LABEL}/pipeline.log`

* Multi-protocol combined setups save to: `output/{EXP_LABEL}/pipeline_multi.log`


---

## Project Structure

```text
├── data/       # Dataset split configurations, protocol lists, and RTTM ground truth
├── setup/      # Environment validation, multi-setup checks, and dataset configurations
├── training/   # Standard and multi-protocol model fine-tuning logic
├── scoring/    # Evaluation metrics, benchmarking, and RTTM generation scripts
├── output/     # Experiment outputs, summary dashboards, and saved model checkpoints
└── utils/      # Database helpers and template generation utility scripts

```

---

## Notes

* Always swap out placeholder references (e.g., `hf_your_token_here`) with real, valid environment strings before starting execution runs.


* Ensure your local system audio configurations match your target dataset sampling rates to prevent processing discrepancies.
