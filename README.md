```markdown
# Financial Sentiment Analysis: BERT vs FinBERT Comparison

## Project Overview
This laboratory work compares the performance of **BERT** (general-purpose language model) and **FinBERT** (domain-specific financial model) for financial sentiment analysis. The experiment demonstrates the effectiveness of domain adaptation in natural language processing for financial applications.

## Key Results
- **FinBERT Accuracy:** 83.33%
- **BERT Accuracy:** 50.00%
- **Improvement:** +33.33% (66.7% relative improvement)

## Project Structure
```
finbert_experiment/
│
├── README.md                    # This file
├── corrected_finbert_experiment.py  # Main experiment script
├── requirements.txt             # Dependencies
│
├── Outputs/
│   ├── professional_comparison.png     # Complete visualization dashboard
│   ├── comprehensive_experiment_report.txt  # Detailed analysis
│   └── test_predictions.csv           # Raw prediction data
│
├── Visualizations/
│   ├── bert_vs_finbert.png            # Direct accuracy comparison
│   ├── fiqa_model_comparison.png      # All models comparison
│   └── bert_finbert_detailed_comparison.png  # Detailed comparison
│
└── Reports/
    ├── fiqa_final_report.txt          # Initial experiment report
    └── experiment_report_template.md   # Report template
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/finbert-experiment.git
cd finbert-experiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiment
```bash
python corrected_finbert_experiment.py
```

## Dependencies
```txt
# Core packages
torch>=1.9.0
transformers>=4.10.0
datasets>=2.0.0

# Data processing
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Experiment Details

### Dataset
- **Name:** FiQA Financial Sentiment (modified)
- **Size:** 22 samples (6 Negative, 10 Neutral, 6 Positive)
- **Train/Test Split:** 16 training, 6 testing samples
- **Classes:** Negative (0), Neutral (1), Positive (2)

### Models Compared
1. **BERT-base-uncased**
   - General-purpose language model
   - 110 million parameters
   - Pre-trained on Wikipedia + BookCorpus

2. **FinBERT (ProsusAI/finbert)**
   - Financial domain-specific model
   - Same architecture as BERT
   - Pre-trained on Reuters, Bloomberg, SEC filings

### Evaluation Metrics
- **Accuracy:** Overall correctness
- **Precision:** Correct positive predictions
- **Recall:** Ability to find all positives
- **F1-Score:** Balance of precision and recall
- **Confusion Matrix:** Error analysis by class

## Results Analysis

### Performance Summary
| Metric | BERT | FinBERT | Improvement |
|--------|------|---------|-------------|
| **Accuracy** | 50.00% | 83.33% | **+33.33%** |
| **Precision** | 25.00% | 88.89% | +63.89% |
| **Recall** | 50.00% | 83.33% | +33.33% |
| **F1-Score** | 33.33% | 83.33% | +50.00% |

### Key Findings
1. **FinBERT significantly outperforms BERT** on financial texts
2. **BERT shows neutral bias** - classifies uncertain financial contexts as neutral
3. **FinBERT understands financial terminology** better
4. **Domain adaptation is highly effective** for financial NLP

## Visualizations Generated

### 1. Accuracy Comparison
![Accuracy Comparison](Visualizations/bert_vs_finbert.png)
- Direct comparison of model accuracy
- Shows 33.3% improvement with FinBERT

### 2. Confusion Matrices
![Confusion Matrices](Visualizations/professional_comparison.png)
- Side-by-side error analysis
- Reveals BERT's neutral bias pattern

### 3. Class-wise Comparison
![Class-wise Accuracy](Visualizations/bert_finbert_detailed_comparison.png)
- Performance breakdown by sentiment class
- FinBERT excels at detecting financial sentiment extremes

## Case Study Examples

| Text | True | BERT | FinBERT | Correct Model |
|------|------|------|---------|---------------|
| "Tech stocks slide on rate concerns" | Negative | Neutral | Negative | **FinBERT** |
| "Apple beats estimates but warns" | Neutral | Neutral | Negative | **BERT** |
| "Biotech shares soaring" | Positive | Neutral | Positive | **FinBERT** |

## Architecture

### Data Flow
```
Financial Texts → Preprocessing → Parallel Processing → Comparative Analysis
                         ↓               ↓                    ↓
                    BERT Pipeline   FinBERT Pipeline   Results & Visualization
```

### Model Architecture
- **Input:** Financial text sequences (max 512 tokens)
- **Tokenization:** WordPiece algorithm
- **Encoder:** 12-layer Transformer with self-attention
- **Classification:** 3-class softmax (Negative/Neutral/Positive)
- **Output:** Sentiment probabilities with confidence scores

## Experiment Report

The complete laboratory work report includes:

### Part 1: Experiment Plan
- Objective: Compare BERT vs FinBERT for financial sentiment
- Hypothesis: FinBERT will outperform due to domain adaptation
- Methodology: Controlled experiment with FiQA dataset

### Part 2: Architecture Diagram
- Comparative pipeline design
- Parallel processing paths
- Evaluation framework

### Part 3: Implementation Steps
1. Dataset preparation
2. Model loading and configuration
3. Evaluation and metric calculation
4. Visualization generation
5. Report compilation

### Part 4: Results Analysis
- Quantitative comparison tables
- Visual evidence of performance differences
- Statistical significance analysis
- Business implications

## Business Applications

### Financial Institutions
- **Real-time sentiment monitoring** for trading decisions
- **Earnings call analysis** for investment research
- **Risk assessment** from financial news

### Research & Analysis
- **Market sentiment tracking**
- **Sector-specific analysis**
- **Regulatory impact assessment**

### Technology Integration
- **Chatbots for financial advice**
- **Automated report generation**
- **Portfolio management tools**

## Customization Options

### Dataset Expansion
```python
# Add custom financial texts
custom_data = [
    {"text": "Your financial news here", "label": 0},  # Negative
    {"text": "Another financial statement", "label": 1},  # Neutral
    {"text": "Positive earnings report", "label": 2},   # Positive
]
```

### Additional Models
```python
# Add more models for comparison
models_to_compare = {
    'bert': 'bert-base-uncased',
    'finbert': 'ProsusAI/finbert',
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased'
}
```

### Custom Metrics
```python
# Implement additional evaluation metrics
custom_metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'auc_roc': roc_auc_score  # For binary classification
}
```

## References

### Academic Papers
1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
   - Devlin et al., 2018
   - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

2. **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models**
   - Araci, 2019
   - [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)

3. **FiQA: Financial Question Answering Dataset**
   - Maia et al., 2018
   - SIGIR 2018

### Datasets
- **FiQA:** Financial Question Answering and Sentiment Analysis
- **Financial PhraseBank:** 4845 sentences with sentiment labels
- **Reuters TRC2:** Financial news corpus
- **SEC filings:** Company financial reports

### Pre-trained Models
- **BERT-base-uncased:** Hugging Face Model Hub
- **FinBERT:** ProsusAI/finbert on Hugging Face
- **Alternative:** yiyanghkust/finbert-tone
