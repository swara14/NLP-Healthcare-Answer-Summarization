# Perspective-Controlled Summarization Pipeline

## Overview

This repository contains a pipeline designed for **perspective-controlled summarization** of healthcare-related community question-answering (CQA) threads. The pipeline consists of two main components:

### 1. Perspective Classifier
- **Model**: Dual-head RoBERTa-based classifier.
- **Function**: Performs multi-label classification to detect the presence of five distinct perspectives in question-answer pairs:
  - INFORMATION  
  - SUGGESTION  
  - EXPERIENCE  
  - QUESTION  
  - CAUSE  

### 2. Perspective-Controlled Generator
- **Model**: BART-based summarization model fine-tuned using LoRA adapters.
- **Function**: Generates concise and relevant summaries, conditioned on the predicted perspectives from the classifier.

---

## Dependencies

Ensure the following Python packages are installed:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `evaluate`
- `tqdm`
- `numpy`
- `bert_score`
- `sacrebleu`

### Installation

Use the following commands to install the dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install datasets
pip install peft
pip install evaluate
pip install tqdm
pip install numpy
pip install bert_score
pip install sacrebleu
```

Pipeline Execution
Ensure all model checkpoints and the test data file are located at the specified paths.

To run the complete pipeline:
```bash
python run_pipeline.py
```
The script performs the following steps:
Perspective Classification
Predicts the perspectives present in each question-answer pair using the classifier.

Perspective-Controlled Summarization
Generates summaries conditioned on each predicted perspective using the generator model.

## Evaluation
If reference summaries are available, the script computes BLEU and BERTScore metrics for each perspective.
## Output
Generated Summaries: Saved to pipeline_2_test_predictions.json
