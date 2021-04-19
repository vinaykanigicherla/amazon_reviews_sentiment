# Sentiment Analysis of Amazon Reviews with Finetuned BERT 

## General Overview 

This repository contains my approach to conducting sentiment analysis on the Amazon Reviews Full Dataset (https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz). It showcases the entire ML workflow I took to approach this problem, including data preprocessing, data downsampling, exploratory data analysis, and modeling. 

The task of "sentiment analysis" conducted in this approach is essentially the prediction of arating (out of 5 stars) for a given Amazon review. I first conducted data preprocessing and downsampling in order to be able to conduct effective  and insightful EDA and also due to the limitations of the computing power of my local machine. Following this, I conduct exploratory data analysis wherein I investigate the structure and relationships within the dataset and generate insights regarding how best to generate embeddings to use to train models for prediction. 

This analysis lead me to understand that the encodings or textual embeddings that I will use to train my model needs to capture not only the sentiment of individual words in a sentence, but also the contextual information of the sentence as a whole. As such, I decided to use a pretrained BERT model as a fature extractor to generate these text embeddings, which I then used as input into a regular classifier neural network which actually predicted the rating of the review.
