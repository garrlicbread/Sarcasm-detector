# Sarcasm Detector using the News Headlines dataset.

# Execution Time measurer
import time 
start = time.time()

# Importing the libraries 
import re
import pandas as pd 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Importing the dataset 
print("Importing the dataset...")

df1 = pd.read_json("C://Desktop/Python/Projects/Sarcasm Detector/Sarcasm_Headlines_Dataset.json", lines = True)
df2 = pd.read_json("C://Desktop/Python/Projects/Sarcasm Detector/Sarcasm_Headlines_Dataset_v2.json", lines = True)
add = [df1, df2]
df = pd.concat(add)

print(f"Size of Dataset = {len(df)}")

# We're only considering statements that have more than 10 words. This will provide more context for the algorithm.
df = df[df.headline.str.count(' ') > 10]

print(f"Size of Dataset = {len(df)}")

y = df.iloc[:, -1].values

print("\nImported.")

# Cleaning the data
print("\nCleaning the dataset...")

ps = PorterStemmer()
clean_data = []
stop_words = set(stopwords.words('english'))

for i in df.index:
    try:
        review = re.sub('[^a-zA-Z]', ' ', str(df['headline'][i]))
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        clean_data.append(review)
        
    except KeyError:
        pass
    
        # # Uncomment the section below this if any errors arise
     
        # df.drop(i)
        # for j in range(i - 10, i):
        #     df.drop(j, axis = 0)
        # for j in range(i, i + 10):
        #     df.drop(j, axis = 0)
            
print("\nCleaned.")

# Creating the bag of words model 
print("\nCreating a bag of words...")
            
cv = CountVectorizer(max_features = 15000)
X = cv.fit_transform(clean_data)
X = X.toarray()

print("\nCreated.")

# Splitting the dataset into test and training set
print("\nSplitting the dataset into training and test set...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print("\nSplit complete.")

# Deleting variables that we don't need anymore
# del df, clean_data, i, stopwords, review, X, y

# Fitting the Random Forest model on the training set and predicting the test set
print("\nTraining the model...")

rf = RandomForestClassifier(n_estimators = 30, 
                            n_jobs = -1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

end = time.time()
exec_time = float((end - start) / 60)
print(f"\nModel Trained. Execution Time: {round(exec_time, 2)} minutes")

# Calculating metrics 
cm = confusion_matrix(y_test, y_pred)
accuracy = (accuracy_score(y_test, y_pred) * 100)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nThe model has been trained and based on the predictions,\n")

print(f"Accuracy = {round(accuracy, 2)}%")
print(f"Precision = {round(precision, 2)}")
print(f"Recall = {round(recall, 2)}")
print(f"F1 score = {round(f1, 2)}")

print("\nThe confusion matrix is: ", "\n", cm)

# Predicting if a user inputted sentence is sarcastic or not 
print("")
print("Enter a sentence and the model will predict if it detects sarcasm or not.")
print("Enter 'Exit' or 'Quit' to break the loop.")
print("")

while True:
    demo = input(r"Enter a sentence: ")
    demo = demo.lower()
    if demo.lower() == "exit" or demo.lower() == "quit":
        break
    demo = re.sub('[^a-zA-Z]', ' ', str(demo))
    demo = demo.split()
    demo = [ps.stem(word) for word in demo if not word in stop_words]
    demo = ' '.join(demo)
    demo = [demo]
    demo_text = cv.transform(demo).toarray()
    demo_pred = rf.predict(demo_text)
    if demo_pred == [1]:
        print("Sarcasm detected.")
    else:
        print("Sarcasm not detected.")
        
    # demo = []

# In theory, this model performs better with the newspaper headline dataset than the reddit dataset
# But in practice, the reddit dataset gives better results.
