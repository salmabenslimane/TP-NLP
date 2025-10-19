# TP2 – BERT Masked Language Modeling
Fine-tuning a BERT model on WikiText-2 with Hugging Face 🤗  
Includes training, evaluation, and qualitative predictions.

## Table of Contents
- [Overview](#overview)
- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Qualitative Predictions](#qualitative-predictions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project focuses on **Masked Language Modeling (MLM)** using **BERT** on the WikiText-2 dataset.  
It demonstrates the full workflow: tokenization, model fine-tuning, evaluation, and generating predictions for masked tokens.

## Project Goals
- Understand **BERT architecture** and masked language modeling.  
- Fine-tune a pre-trained BERT model for a specific NLP task.  
- Evaluate model performance on WikiText-2.  
- Generate qualitative predictions to inspect model understanding.

## Dataset
We use the [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset:  
- Open-source collection of English Wikipedia articles  
- Commonly used for language modeling tasks  
- Already split into training, validation, and test sets  

## Installation
Install dependencies using `pip`:

```bash
pip install torch transformers datasets scikit-learn pandas matplotlib
Usage
```

## Clone the repository:

git clone https://github.com/salmabenslimane/TP-NLP.git
cd TP-NLP

-Open the Jupyter notebook TP2_BERT.ipynb.

-Run cells sequentially to reproduce preprocessing, training, and evaluation.

## Training

-Use BertForMaskedLM from Hugging Face.

-Tokenize dataset with BertTokenizer.

-Fine-tune on WikiText-2 using Trainer API or custom training loop.

-Adjust hyperparameters: learning rate, batch size, epochs.

## Evaluation

-Compute loss on validation set.

-Optionally compute perplexity as a measure of model performance.

-Inspect masked token predictions for qualitative evaluation.

## Qualitative Predictions

-Mask words in sentences and use the model to predict missing tokens.

-Compare predictions with ground truth for interpretability.

-Useful for understanding the linguistic capabilities of BERT.

## Results

-Include plots of training/validation loss over epochs.

-Example masked token predictions.

-Observations about model strengths and weaknesses.

## Contributing

Salma B. & ilyas D. 

## License

This project is released under the MIT License.
