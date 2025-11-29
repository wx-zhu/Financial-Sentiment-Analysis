# PSO-Optimized DistilBERT for Financial Sentiment Analysis

**STATS 507 Final Project - Fall 2024**  
**Author:** Wenxuan Zhu  
**University of Michigan, Department of Biostatistics**

## Overview

This project investigates the application of Particle Swarm Optimization (PSO) for hyperparameter tuning of DistilBERT in financial sentiment classification. We systematically compare baseline DistilBERT performance against PSO-optimized configurations using the Financial PhraseBank dataset.

## Key Findings

- **High-Agreement Data (AllAgree Subset):** PSO achieves modest improvement (95.14% → 96.03%)
- **Complete Dataset (All Agreement Levels):** Baseline outperforms PSO (83.81% vs 81.65%)
- **Key Insight:** Well-established default hyperparameters often represent near-optimal configurations for mature NLP architectures

## Project Structure

```
.
├── data/                           # Dataset directory
│   └── financial_phrasebank/       # Financial PhraseBank dataset
├── src/                            # Source code
│   ├── financial_sentiment_pso_complete.ipynb    # AllAgree subset experiments
│   └── financial_sentiment_pso_FINAL.ipynb       # Complete dataset experiments
└── README.md
```

## Dataset

**Financial PhraseBank** (Malo et al., 2014)
- 4,846 sentences from financial news
- Annotated by 16 domain experts
- Three sentiment classes: positive, neutral, negative
- Stratified by annotator agreement levels (50%, 66%, 75%, 100%)

We conduct experiments on two configurations:
1. **AllAgree Subset:** 2,264 sentences with 100% annotator agreement
2. **Complete Dataset:** All 4,846 sentences with varying agreement levels

## Methodology

### Model Architecture
- **Base Model:** DistilBERT-base-uncased
- **Task:** 3-class sentiment classification (positive/neutral/negative)
- **Tokenization:** WordPiece, max length 128

### Baseline Configuration
- Learning rate: 2×10⁻⁵
- Dropout: 0.3
- Batch size: 16
- Optimizer: AdamW with linear warmup

### PSO Optimization
- **Optimized Hyperparameters:** Learning rate, dropout rate, classification head hidden size
- **PSO Configuration:** 
  - Inertia weight: 0.9
  - Cognitive coefficient: 2.0
  - Social coefficient: 2.0
- **AllAgree:** 5 particles × 3 iterations (15 evaluations)
- **Complete Dataset:** 10 particles × 10 iterations (~55 evaluations with early stopping)

## Results Summary

### Experiment 1: AllAgree Subset (2,264 sentences)

| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| Baseline DistilBERT | 95.14% | 0.209 |
| PSO-Optimized | 96.03% | 0.242 |
| **Improvement** | **+0.89 pp** | — |

**Optimal Hyperparameters Found:**
- Learning Rate: 1.07×10⁻⁴
- Dropout: 0.312
- Hidden Size: 417

### Experiment 2: Complete Dataset (4,846 sentences)

| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| Baseline DistilBERT | 83.81% | 0.810 |
| PSO-Optimized | 81.65% | 0.911 |
| PSO + Regularization | 82.37% | 0.872 |
| **Change (PSO vs Baseline)** | **-2.16 pp** | — |

**Optimal Hyperparameters Found:**
- Learning Rate: 1.02×10⁻⁴
- Dropout: 0.211
- Hidden Size: 386


## Key Takeaways

1. **Dataset Quality Matters:** PSO is more effective on clean, high-agreement data
2. **Default Hyperparameters Are Robust:** Standard BERT fine-tuning defaults are particularly robust to label noise
3. **Computational Cost vs. Benefit:** For mature architectures like DistilBERT, marginal gains from optimization may not justify computational cost when training data contains label noise
4. **Regularization Helps (Partially):** Adding stronger regularization (dropout 0.4, weight decay 0.01) partially recovers performance but still underperforms baseline

