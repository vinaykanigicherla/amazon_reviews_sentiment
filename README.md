# Sentiment Analysis of Amazon Reviews with Finetuned BERT  
 
## General Overview  
 
This repository contains my approach to conducting sentiment analysis on the Amazon Reviews Full Dataset (https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz). It showcases the entire ML workflow I took to approach this problem, including data preprocessing, data downsampling, exploratory data analysis, and modeling.  
 
The task of "sentiment analysis" conducted in this approach is essentially the prediction of arating (out of 5 stars) for a given Amazon review. I first conducted data preprocessing and downsampling in order to be able to conduct effective  and insightful EDA and also due to the limitations of the computing power of my local machine. Following this, I conduct exploratory data analysis wherein I investigate the structure and relationships within the dataset and generate insights regarding how best to generate embeddings to use to train models for prediction.  
 
This analysis lead me to understand that the encodings or textual embeddings that I will use to train my model needs to capture not only the sentiment of individual words in a sentence, but also the contextual information of the sentence as a whole. As such, I decided to use a pretrained BERT model as a fature extractor to generate these text embeddings, which I then used as input into a regular classifier neural network which actually predicted the rating of the review.

The order in which to run the above Jupyter Notebooks is:
1. data_sampling_preprocessing.ipynb
2. eda.ipynb
3. bert_embeddings.ipynb
4. classical_models.ipynb
5. bert_transfer_learning.ipynb

## Description of Project Stages
### Downsampling and Preprocessing
Due to the enormous size of the entire dataset, I could not work with the entire dataset on my local machine as it was not powerful enough to do so. Consequently, I had to downsample the data. When doing so, I took extra care in ensuring that distribution of labels after downsampling is preserved and that there were an equal number of data points for each label.

Prior to this downsampling, I combined both the review title and review body into a single string and preprocessed the entire data in order to be able to effectively use it during the exploratory data analysis stage of the project. The preprocessing fucntions I had to apply on the textual data were:
- Stripping HTML Tags (gpp.strip_tags)
- Removing all Punctuation (gpp.strip_punctuation)
- Removing all extra whitespaces (gpp.strip_multiple_whitespaces)
- Removing all numerics (gpp.strip_numeric)
- Removing stopwords(gpp.remove_stopwords)
- Removing words shorter than 3 letters (gpp.strip_short)


### Exploratory Data Analysis

In the EDA notebook, I discovered that while the word level sentiment of the words in the reviews were correlated with the ratings, using these simple sentiment polarity values would not be enough to achieve good accuracy on the dataset. After inspecting the most common words in the reviews and ordering them by polarity, I realized that it was necessary to capture the contextual information of the review in its encoding/embedding

### Generating BERT Embeddings

This was the stage wherein I actually generated the data that would be used to train the predictive models. I generated the text embeddings using a pretrained BERT model from Tensorflow Hub finetuned specifically for sentiment analysis.

### Classical ML Models Baseline

Once I generated the embeddings, I attempted to apply classicaly ML models to the problem whose score the neural network approach would need to beat. I fitted the RandomForest, Linear Support Vector Classifier, and Logistic Regression models to the train data and conducted hyperparameter tuning for each. I chose these models as they were ones that supported multiclass classification by default. Out of these, the Logistic Regression model scored the highest with a 20.8% accuracy. 

### BERT Transfer Learning Approach

In the final notebook, I created a neural network classifier and stacked it on top of the BERT encoder and trained it on the downsampled data. At first I simply added a single dense layer on top of the BERT encoder. However, during training, I realized that this model's validation accuracy stalled at around 40%. In order to improve the model, I added a two layer classifier instead and also made sure to add L2 weight regularization and dropout in order to combat overfitting. This approach achieved a 53% accuracy, differing from the SOTA benchmark for the Amazon Review Dataset by 12%.  

## Explanation of Key Decisions
- I chose to use the BERT model for transfer learning instead of ULM-FiT as my research indicated that BERT performed better for sentiment analysis tasks, especially when using the BERT Expert model
- I chose not to apply any form of data augmentation to my data as I alreaady had a surplus of data which I could not use due to my computing power limitations  
- I chose to only train the weight of my classifier neural network and not the weights of BERT encoder itself (due to which there was no further finetuning of the encoder specific to this dataset) due to computational limitations  
- I chose to make my classifier model only two layers deep as the addition of more layers would cause further overfitting (which I was already facing and combatting)

## Limitations and Improvements

This approach has a number of limitations:
- I was not able to use the entirety of the dataset, due to which my model's predictions may not generalize well to the entire dataset
- I did not implement data augmentation on the dataset which could have further improved the model's generalizability and accuracy

Possible improvements:
- Training on the complete dataset and augmented data would improve performance (requires better computational resources)
- Using AdamW (Adam with Weight Decay) optimizer instead of the regular Adam optimizers
- Better weight initialization for classifier model
- Training for more epochs with a larger batch size (requires better computational resources)
