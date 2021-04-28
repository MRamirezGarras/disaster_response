# import libraries
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score
from statistics import mean
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


def scores(pred, test):
    """
    Prints f1 score, recall and precision for each column.
    Prints mean f1 score, recall and precision for all columns
    Saves all values in a dataframe

    INPUT:
    pred: predictions from a model
    test: real values

    OUTPUT:
    result: dataframe with the f1 score, recall and precision for each column
    """

    # Creates lists to save the f1 score, recall and precision for each column
    f1scores = []
    recallscores = []
    precisionscores = []

    for col in test:
        print(col)
        f1score = f1_score(test[col], pred[col], average="weighted")
        f1scores.append(f1score)
        print("F1 score: ", f1score)
        recallscore = recall_score(test[col], pred[col], average="weighted")
        recallscores.append(recallscore)
        print("Recall score: ", recallscore)
        precisionscore = precision_score(test[col], pred[col], average="weighted")
        precisionscores.append(precisionscore)
        print("Precision score: ", precisionscore)

        result = pd.DataFrame(list(zip(test.columns, f1scores, recallscores, precisionscores)),
        columns = ["Column", "F1_score", "Recall", "Precision"])

        print("Mean F1 score svm model:", mean(f1scores))
        print("Mean recall svm simple model:", mean(recallscores))
        print("Mean precision score svm model:", mean(precisionscores))

    return (result)

#####SVM model############

#Split data in train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=17)

pipeline_svm = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('svm', MultiOutputClassifier(SVC()))
    ])

pipeline_svm.fit(x_train, y_train)#Fit the model
y_pred = pipeline_svm.predict(x_test)#Predict using the test set
y_pred = pd.DataFrame.from_records(y_pred)#Store the predictions in a dataframe
y_pred.columns = y_test.columns


svm_model_result = scores(y_pred, y_test)

#Select 10% of the samples to search for the best parameters
#It is necessary to work with a reduced number of samples because it could take too long
x_train_sample, x_test_sample, y_train_sample, y_test_sample = train_test_split(X, Y, random_state=17, train_size=0.1)

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
best_parameters = model.best_params_
print("Best Parameters:", best_parameters)

#Create pipeline for the parameters selected
pipeline_svm_tuned = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, ngram_range=(1,1), max_features=None)),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('svm', MultiOutputClassifier(SVC(kernel="linear")))
    ])

pipeline_svm_tuned = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=best_parameters["vect__max_df"], ngram_range= best_parameters["vect__ngram_range"],
                                 max_features=best_parameters["vect__max_features"])),
        ('tfidf', TfidfTransformer(use_idf=best_parameters["tfidf__use_idf"])),
        ('svm', MultiOutputClassifier(SVC(kernel = best_parameters["svm__estimator__kernel"])))
    ])


#Train the model
pipeline_svm_tuned.fit(x_train, y_train)

y_pred = pipeline_svm_tuned.predict(x_test)
y_pred = pd.DataFrame.from_records(y_pred)
y_pred.columns = y_test.columns

tuned_svm_model_result = scores(y_pred, y_test)


#Save the model
pkl_filename = "svm_model_tuned.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pipeline_svm_tuned, file)