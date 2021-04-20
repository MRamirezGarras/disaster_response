# import libraries
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score
from statistics import mean
from sklearn.ensemble import RandomForestClassifier
import pickle

#install punkt
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# load data from database
engine = create_engine('sqlite:///database_messages.db').connect()
df = pd.read_sql_table("database_messages", engine)

#Select values to train the model
X = df["message"]
Y = df.iloc[:, 4:40]

#Check the values for each column
for col in Y:
    print(col, ":", Y[col].unique())

#The column "child_alone" has only one type of value, so I cannot be predicted. I remove it
Y.drop("child_alone", axis = 1, inplace = True)

def tokenize(text):
    """
    This function get a piece of text and returns a list of tokens for text classification
    """
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', " ", text)#Remove unusual symbols
    tokens = word_tokenize(text)
    words = [w for w in tokens if w not in stopwords.words("english")]#Remove stopwords
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return(clean_tokens)

###Model with Random Forest##########

pipeline_rf = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

#Split data in train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=17)

pipeline_rf.fit(x_train, y_train)#Fit the model
y_pred = pipeline_rf.predict(x_test)#Predict using the test set
y_pred = pd.DataFrame.from_records(y_pred)#Store the predictions in a dataframe
y_pred.columns = y_test.columns

#Creates lists to save the f1 score, recall and precision for each column
f1scores_rf = []
recallscores_rf = []
precisionscores_rf = []

for col in y_test:
    f1scores_rf.append(f1_score(y_test[col], y_pred[col], average="weighted"))
    recallscores_rf.append(recall_score(y_test[col], y_pred[col], average="weighted"))
    precisionscores_rf.append(precision_score(y_test[col], y_pred[col], average="weighted"))

rf_model_result = pd.DataFrame(list(zip(y_test.columns, f1scores_rf, recallscores_rf, precisionscores_rf)),
                                  columns= ["Column", "F1_score", "Recall", "Precision"])


def tune_model_rf():
    """
    Returns a model with a list of parameters to feed to GridSearchCV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator_n_estimators': [50, 100, 200],
        'clf__estimator_min_samples_split': [2, 3, 4],
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

#I decreased the size of the train size because the fit() function run for too long
x_train_sample, x_test_sample, y_train_sample, y_test_sample = train_test_split(X, Y, random_state=17, train_size=0.1)
model = tune_model_rf()
print("Model built")
model.fit(x_train_sample, y_train_sample)
print("Model fitted")
y_pred = model.predict(x_test_sample)
print("Best Parameters:", model.best_params_)

#Create pipeline for the parameters selected
pipeline_rf_tuned = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, ngram_range=((1,1), (1,2)), max_features=None)),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('svm', MultiOutputClassifier(RandomForestClassifier(min_samples_split = 4, n_estimators = 100)))
    ])

#Train the model
pipeline_rf_tuned.fit(x_train, y_train)

y_pred = pipeline_rf_tuned.predict(x_test)
y_pred = pd.DataFrame.from_records(y_pred)
y_pred.columns = y_test.columns

f1scores_rf_tuned = []
recallscores_rf_tuned = []
precisionscores_rf_tuned = []
for col in y_test:
    f1scores_rf_tuned.append(f1_score(y_test[col], y_pred[col], average="weighted"))
    recallscores_rf_tuned.append(recall_score(y_test[col], y_pred[col], average="weighted"))
    precisionscores_rf_tuned.append(precision_score(y_test[col], y_pred[col], average="weighted"))

rf_tuned_result = pd.DataFrame(list(zip(y_test.columns, f1scores_rf_tuned, recallscores_rf_tuned, precisionscores_rf_tuned)),
                                      columns=["Column", "F1_score", "Recall", "Precision"])

print("Mean F1 score rf tuned model:", mean(f1scores_rf_tuned))
print("Mean recall score rf tuned model:", mean(recallscores_rf_tuned))
print("Mean precision score rf tuned model:", mean(precisionscores_rf_tuned))




#####Try SVM model############

pipeline_svm = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('svm', MultiOutputClassifier(SVC()))
    ])

pipeline_svm.fit(x_train, y_train)#Fit the model
y_pred = pipeline_svm.predict(x_test)#Predict using the test set
y_pred = pd.DataFrame.from_records(y_pred)#Store the predictions in a dataframe
y_pred.columns = y_test.columns

#Creates lists to save the f1 score, recall and precision for each column
f1scores_svm = []
recallscores_svm = []
precisionscores_svm = []

for col in y_test:
    f1scores_svm.append(f1_score(y_test[col], y_pred[col], average="weighted"))
    recallscores_svm.append(recall_score(y_test[col], y_pred[col], average="weighted"))
    precisionscores_svm.append(precision_score(y_test[col], y_pred[col], average="weighted"))

first_model_result = pd.DataFrame(list(zip(y_test.columns, f1scores_svm, recallscores_svm, precisionscores_svm)),
                                  columns= ["Column", "F1_score", "Recall", "Precision"])

print("Mean F1 score svm model:", mean(f1scores_svm))
print("Mean recall svm simple model:", mean(recallscores_svm))
print("Mean precision score svm model:", mean(precisionscores_svm))


def tune_model_svm():
    """
    Returns a model with a list of parameters to select the best with GridSearchCV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('svm', MultiOutputClassifier(SVC())),
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'svm__estimator__kernel': ("linear", "rbf", "sigmoid"),
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


model = tune_model_svm()
print("Model built")
model.fit(x_train_sample, y_train_sample)
print("Model fitted")
y_pred = model.predict(x_test_sample)
print("Best Parameters:", model.best_params_)

#Create pipeline for the parameters selected
pipeline_svm_tuned = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, ngram_range=(1,1), max_features=None)),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('svm', MultiOutputClassifier(SVC(kernel="linear")))
    ])

#Train the model
pipeline_svm_tuned.fit(x_train, y_train)

y_pred = pipeline_svm_tuned.predict(x_test)
y_pred = pd.DataFrame.from_records(y_pred)
y_pred.columns = y_test.columns

f1scores_svm_tuned = []
recallscores_svm_tuned = []
precisionscores_svm_tuned = []
for col in y_test:
    f1scores_svm_tuned.append(f1_score(y_test[col], y_pred[col], average="weighted"))
    recallscores_svm_tuned.append(recall_score(y_test[col], y_pred[col], average="weighted"))
    precisionscores_svm_tuned.append(precision_score(y_test[col], y_pred[col], average="weighted"))

svm_tuned_result = pd.DataFrame(list(zip(y_test.columns, f1scores_svm_tuned, recallscores_svm_tuned, precisionscores_svm_tuned)),
                                      columns=["Column", "F1_score", "Recall", "Precision"])

print("Mean F1 score adjusted model:", mean(f1scores_svm_tuned))
print("Mean recall score adjusted model:", mean(recallscores_svm_tuned))
print("Mean precision score adjusted model:", mean(precisionscores_svm_tuned))

#Save the model
pkl_filename = "svm_model_tuned.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pipeline_svm_tuned, file)