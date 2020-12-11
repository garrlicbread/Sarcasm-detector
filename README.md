# Sarcasm-detection-using-NLP

Natural Language Processing (NLP) is a subfield of Artifical Intelligence that helps a computer read, decipher, understand, and make sense of the human languages in a manner that is valuable.

## Problem: 
This project tries to predict if a given statement is sarcastic in nature or not. Sarcasm is the use of irony to mock or convey contempt in a subtle or obvious manner.

## Dataset: 
There are two different scripts that load two different datasets. Both can be found on kaggle and their links are attached below. The first script loads a dataset containing sarcastic and genuine remarks from Reddit. The second script loads a dataset containing sarcastic and genuine news headlines. Although, the news headlines dataset leads the model to give a higher accuracy, in practise, the script loading the reddit dataset performs slightly better. 

## Process: 
Both of the scripts begin by pre-processesing the raw text by calling the necessary classes. They then convert the remaining words into a huge vector containing 15,000 of the most frequent words found in sarcastic and non-sarcastic comments/headlines. After splitting the data into the training and test set, it applies the Random Forest classifier on the vector and prints metrics like the confusion matrix, accuracy, precision, recall and the F1 score. Finally, it runs a loop that asks the user to input a sentence which is preprocessed manually and then a statement is printed revealing if the model detects sarcasm in the sentence or not. 

## Outputs:

Given below are the testing metrics achieved by both the scripts. 'Reddit' refers to the script that loads the Reddit comments dataset. 'Headlines' refers to the script that loads the news headlines dataset.

| Metrics (Reddit) | Values  | Metrics (News Headlines) | Values | 
| ---------------- | ------- | ------------------------ | ------ |
| Accuracy (in %)  | 66.8    | Accuracy (in %)          | 79.7   |
| Precision        | 0.67    | Precision                | 0.82   |
| Recall           | 0.65    | Recall                   | 0.77   |
| F1 Score         | 0.66    | F1 Score                 | 0.80   |

## Example predictions:

Both the scripts perform similar in practise with a few false positives here and there. 

### Let's look at a couple of examples that the model detects correctly.

&gt;&gt; Well, this day was a total waste of a good outfit. (Input sentence)

&gt; Sarcasm detected. (Prediction)

&gt;&gt; Tell me something I don’t know.

&gt; Sarcasm detected.

&gt;&gt; It’s so thoughtful for the teacher to give us all this homework right before Spring Break.

&gt; Sarcasm detected.

&gt;&gt; Whisper my favorite words: “It’s free of charge.”

&gt; Sarcasm detected.

&gt;&gt; Listening to the news! Again? Well, it changes every day, you see.

&gt; Sarcasm detected.

### Now let's look at a couple of examples that the model fails to detect 

&gt;&gt; Either you haven't taken a bath for days or you have a questionable choice in perfumes.

&gt; Sarcasm not detected.

&gt;&gt; Do you really think this country is going to elect an orange guy with a funny name to be president of the US?

&gt; Sarcasm not detected.

&gt;&gt; When I hear you speak I feel sorry for my brain cells.

&gt; Sarcasm not detected.

## Conclusion:
The accuracies that these models achieve is between 66% and 79% which might not look very impressive but then we must consider the fact that this program only helps a computer to classify based on raw text. In most cases, humans detect sarcasm based on the tone of the speaker which is a privilege out of the scope of this project. Nevertheless, both the scripts perform on par when fed with one liner sarcastic remarks. To get the best out of this model, the input sentences are recommended to be equal to or more than ten words.

References:

1) News Headlines dataset : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
2) Reddit Comments dataset: https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit
3) One liner sarcastic examples: https://examples.yourdictionary.com/examples-of-sarcasm.html
4) More examples: https://www.examples.com/education/sarcasm-examples.html
