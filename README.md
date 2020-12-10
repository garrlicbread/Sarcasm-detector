# Sarcasm-detection-using-NLP

Natural Language Processing (NLP) is a subfield of Artifical Intelligence that helps a computer read, decipher, understand, and make sense of the human languages in a manner that is valuable.

## Problem: 
Normally, a beginner project in NLP involves classifying sentiments of a review, tweet about a movie, book, restaurant or stock market. Instead of that, this project tries to predict if a given statement is sarcastic in nature or not. 

## Dataset: 
There are two different scripts that load two different datasets. Both can be found on kaggle and their links are attached below. The first script loads a dataset containing sarcastic and genuine remarks from Reddit. The second script loads a dataset containing sarcastic and genuine news headlines. Although, the news headlines dataset leads the model to give a higher accuracy, in practise, the reddit dataset performs better. 

## Process: 
The script begins by pre-processesing the raw text by calling the neceassary classes. It then converts the remaining words into a huge vector containing 15,000 of the most frequent words found in sarcastic and non-sarcastic sentences. After splitting the data into the training and test set, it applies the Random Forest classifier on the vector and prints metrics like the confusion matrix, accuracy, precision, recall and the F1 score. Finally, it runs a while loop and asks the user to input a sentence which is preprocessed and the output prints if the model detects sarcasm in the sentence or not. 

## Outputs:

Given below are the metrics achieved by both the scripts. Reddit refers to the script that loads the comments taken from Reddit. Headlines refers to the script that loads the news headlines dataset.

| Metrics (Reddit) | Values  | Metrics (News Headlines) | Values | 
| ---------------- | ------- | ------------------------ | ------ |
| Accuracy         | 66.88%  | Accuracy                 | 79.7%  |
| Precision        | 0.67    | Precision                | 0.82   |
| Recall           | 0.65    | Recall                   | 0.77   |
| F1 Score         | 0.66    | F1 Score                 | 0.80   |

## Example predictions:


## Conclusion:
The accuracies that these models achieve is between 66% and 79% which might not look very impressive but then we must consider the fact that this program only helps a computer to classify based on raw text. In most cases, humans detect sarcasm based on the tone of the speaker which is a privilege out of the scope of this project.

References:

1) News Headlines dataset : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
2) Reddit Comments dataset: https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit
3) Sarcasm examples: https://examples.yourdictionary.com/examples-of-sarcasm.html
