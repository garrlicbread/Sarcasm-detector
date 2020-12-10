# Sarcasm-detection-using-NLP

Natural Language Processing (NLP) is a subfield of Artifical Intelligence that helps a computer read, decipher, understand, and make sense of the human languages in a manner that is valuable.

## Problem: 
This project tries to predict if a given statement is sarcastic in nature or not. Sarcasm is the use of irony to mock or convey contempt in a subtle or obvious manner.

## Dataset: 
There are two different scripts that load two different datasets. Both can be found on kaggle and their links are attached below. The first script loads a dataset containing sarcastic and genuine remarks from Reddit. The second script loads a dataset containing sarcastic and genuine news headlines. Although, the news headlines dataset leads the model to give a higher accuracy, in practise, the script loading the reddit dataset performs slightly better. 

## Process: 
Both of the scripts begin by pre-processesing the raw text by calling the neceassary classes. They then convert the remaining words into a huge vector containing 15,000 of the most frequent words found in sarcastic and non-sarcastic comments/headlines. After splitting the data into the training and test set, it applies the Random Forest classifier on the vector and prints metrics like the confusion matrix, accuracy, precision, recall and the F1 score. Finally, it runs a loop that asks the user to input a sentence which is preprocessed manually and then a statement is printed revealing if the model detects sarcasm in the sentence or not. 

## Outputs:

Given below are the testing metrics achieved by both the scripts. 'Reddit' refers to the script that loads the Reddit comments dataet. 'Headlines' refers to the script that loads the news headlines dataset.

| Metrics (Reddit) | Values  | Metrics (News Headlines) | Values | 
| ---------------- | ------- | ------------------------ | ------ |
| Accuracy (in %)  | 66.8    | Accuracy (in %)          | 79.7   |
| Precision        | 0.67    | Precision                | 0.82   |
| Recall           | 0.65    | Recall                   | 0.77   |
| F1 Score         | 0.66    | F1 Score                 | 0.80   |

## Example predictions:

Both the scripts perform similar in practise with a few false positives here and there. Let's look at a couple of examples. 

> Input sentence
>  Prediction

<addr>

>> Well, this day was a total waste of a good outfit.
> Sarcasm detected.

>> Tell me something I don’t know.
> Sarcasm detected.



## Conclusion:
The accuracies that these models achieve is between 66% and 79% which might not look very impressive but then we must consider the fact that this program only helps a computer to classify based on raw text. In most cases, humans detect sarcasm based on the tone of the speaker which is a privilege out of the scope of this project.

References:

1) News Headlines dataset : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
2) Reddit Comments dataset: https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit
3) One liner sarcastic examples: https://examples.yourdictionary.com/examples-of-sarcasm.html
4) 
