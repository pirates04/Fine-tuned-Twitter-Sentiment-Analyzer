# 🐦 Fine-Tuned Twitter Sentiment Analyzer

This project builds a **domain-specific sentiment analyzer** for tweets by fine-tuning a **BERT transformer model** on the **TweetEval Sentiment dataset**.  
The model classifies tweets into **Positive, Negative, or Neutral** with high accuracy, making it useful for real-world applications.

## 🎯 Objective
To create a state-of-the-art sentiment classification model for social media data that can be used for:
- Brand monitoring
- Customer feedback analysis
- Social media trend detection

## 🛠️ Tech Stack
- **Python** – Core programming language  
- **Hugging Face Transformers** – Pre-trained BERT model fine-tuning  
- **Datasets & Evaluate** – Dataset loading and metrics  
- **PyTorch** – Deep learning backend  
- **Scikit-learn** – Classification report and evaluation  
- **Matplotlib/Seaborn** – Visualizations (EDA)  
- **BERT Model** - Transformers

## 📊 Results
| Model              | Accuracy | Weighted F1 |
|-------------------|---------|-------------|
| Fine-tuned BERT   | **~89%** | **~0.88** |

The fine-tuned BERT model shows **strong performance** for tweet-level sentiment classification.

## 🚀 Key Features
- Preprocessing pipeline for cleaning Twitter data (removing mentions, links, special characters).
- Fine-tuning of `bert-base-uncased` on TweetEval Sentiment task.
- Detailed evaluation using Accuracy, Precision, Recall, and F1-score.
- Ready-to-use inference script for predicting sentiment on custom tweets.


