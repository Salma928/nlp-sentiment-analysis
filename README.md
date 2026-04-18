# Sentiment Analysis with XLM-RoBERTa

Fine-tuned XLM-RoBERTa model for multilingual sentiment analysis, with an interactive Streamlit demo.

## Overview

This project fine-tunes [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) on the SST-2 dataset to classify text as **positive** or **negative**. Thanks to XLM-RoBERTa's multilingual pretraining, the model generalizes across 100 languages.

## Demo

Run the app locally:

```bash
streamlit run app.py
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| XLM-RoBERTa | Pretrained multilingual transformer |
| HuggingFace Transformers | Model fine-tuning & inference |
| HuggingFace Datasets | SST-2 dataset loading |
| PyTorch | Deep learning framework |
| Scikit-learn | Evaluation metrics |
| Streamlit | Interactive demo |

## Dataset

**SST-2** (Stanford Sentiment Treebank) — 67,349 English movie review sentences labeled as positive or negative. Available on [HuggingFace](https://huggingface.co/datasets/sst2).

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~75% |
| F1-score | ~75% |

> Trained on 2,000 samples / 1 epoch on CPU for reproducibility.

## Project Structure
├── train.py       # Fine-tuning script
├── app.py         # Streamlit demo
├── model/         # Saved model (not tracked by git)
└── README.md

## Installation

```bash
git clone https://github.com/Salma928/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
python -m venv venv
source venv/bin/activate
pip install transformers==4.40.0 datasets torch scikit-learn streamlit accelerate
python train.py
streamlit run app.py
```

## Author

**Salma Bentahar Alaoui** — M2 Mesure et Traitement de l'Information, Université de Lorraine, France, et Ingénieure d'état lauréate de l'Ecole Nationale des Sciences Appliquées de Tanger, Maroc.
AI/ML Engineer seeking CDI from October 2026