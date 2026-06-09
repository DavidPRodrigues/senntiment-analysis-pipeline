# Customer Review Sentiment Analysis
An end-to-end pipeline that turns thousands of unstructured customer reviews into actionable sentiment insights and business recommendations.

## Problem
Companies receive far mroe reviews than anyone can read. This project classifies review sentiment at scale and surfaces what customers actually complain about - so the business knows what to fix first.

## Stack
- Python (pandas, scikit-learn, NLTK/Hugging Face)
- Streamlit (dashboard)
- Deployed on Streamlit community Cloud

## Project strucutre 
- 'src/' - data cleaning, preprocessing, modelling, evaluation
- 'notebooks/' - exploratory analysis
- 'dashboard/' - Streamlit app
- 'reports/' - business insight memo

## Status
In progress

## Results
Accuracy: 87.6%

## Run

python src/data_loading.py
python src/data_cleaning.py
python src/preprocessing.py
python src/train_model.py
python src/predict.py

