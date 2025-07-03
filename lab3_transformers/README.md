# Lab 3: Transformers & HuggingFace Ecosystem

This laboratory explores how to use the HuggingFace ecosystem to adapt and fine-tune Transformer models for new tasks, focusing on sentiment analysis with DistilBERT and efficient fine-tuning techniques.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Lab Structure](#lab-structure)
  - [Exercise 1: Sentiment Analysis (Warm Up)](#exercise-1-sentiment-analysis-warm-up)
  - [Exercise 2: Fine-tuning DistilBERT](#exercise-2-fine-tuning-distilbert)
  - [Exercise 3: Efficient Fine-tuning & Advanced Tasks](#exercise-3-efficient-fine-tuning--advanced-tasks)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Results & Observations](#results--observations)
- [References](#references)

---

## Overview

The goal of this lab is to:
- Learn how to use HuggingFace Datasets and Transformers libraries.
- Extract features from a pre-trained DistilBERT model and use them for sentiment classification.
- Fine-tune DistilBERT for improved performance.
- Explore efficient fine-tuning methods (e.g., LoRA/PEFT).
- Optionally, experiment with CLIP or other models.

---

## Dataset

- **Source:** [Cornell Rotten Tomatoes movie review dataset](https://huggingface.co/datasets/rotten_tomatoes)
- **Splits:** `train` (8530), `validation` (1066), `test` (1066)
- **Structure:** Each example is a dict with `text` (movie review) and `label` (0=negative, 1=positive).

---

## Lab Structure

### Exercise 1: Sentiment Analysis (Warm Up)

1. **Dataset Exploration**
   - Load the Rotten Tomatoes dataset using HuggingFace Datasets.
   - Inspect available splits and data structure.

2. **Feature Extraction with DistilBERT**
   - Load a pre-trained DistilBERT model and tokenizer.
   - Tokenize sentences and extract `[CLS]` token representations from the last hidden state.
   - Use these representations as features for classification.

3. **Baseline Classifier**
   - Train a linear SVM classifier on the extracted features.
   - Evaluate accuracy on validation and test splits.
   - This serves as a strong baseline for comparison.

### Exercise 2: Fine-tuning DistilBERT

1. **Tokenization**
   - Tokenize the dataset splits with the DistilBERT tokenizer, ensuring each example has `input_ids` and `attention_mask`.

2. **Model Preparation**
   - Load `AutoModelForSequenceClassification` with DistilBERT and a classification head.

3. **Training with HuggingFace Trainer**
   - Set up a `Trainer` with data collator, training arguments, and a custom metric function (accuracy, precision, recall, F1).
   - Fine-tune the model on the training split and evaluate on validation and test splits.
   - Save the fine-tuned model and tokenizer.

4. **Comparison**
   - Compare the SVM baseline with the fine-tuned DistilBERT using a summary table.

### Exercise 3: Efficient Fine-tuning & Advanced Tasks

1. **Efficient Fine-tuning with LoRA/PEFT**
   - Use the [PEFT library](https://huggingface.co/docs/peft/en/index) to apply LoRA to DistilBERT.
   - Fine-tune only a small subset of parameters for efficiency.
   - Save LoRA weights and metrics.

---

## Dependencies

- Python 3.7+
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- datasets (`pip install datasets`)
- transformers (`pip install transformers`)
- peft, accelerate, bitsandbytes (for LoRA/efficient fine-tuning)
- google.colab (if using Google Drive for saving)
- fsspec, huggingface_hub

---

## Results & Observations

- **Baseline (SVM on DistilBERT CLS features):**
  - Validation accuracy: *[insert value from notebook output]*
  - Test accuracy: *[insert value from notebook output]*

- **Fine-tuned DistilBERT:**
  - Test accuracy: *[insert value from notebook output]*
  - Precision, Recall, F1: *[insert values]*

- **Efficient Fine-tuning (LoRA):**
  - Comparable performance with much fewer trainable parameters.

- **General Observations:**
  - Fine-tuning the full model yields better results than using static features.
  - Efficient fine-tuning methods (like LoRA) allow for fast adaptation with minimal compute.
  - HuggingFace abstractions make it easy to experiment with state-of-the-art models.

---

## References

- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/en/index)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/en/index)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/en/index)
- [DistilBERT Model Card](https://huggingface.co/distilbert-base-uncased)


---
