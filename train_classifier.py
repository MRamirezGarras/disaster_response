import sys
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

# install punkt
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///database_messages.db').connect()
    df = pd.read_sql_table("database_messages", engine)
    # Split values in two variables
    X = df["message"]
    Y = df.iloc[:, 4:40]
    categories = Y.columns

    return (X, Y, categories)


def tokenize(text):
    """
    This function get a piece of text and returns a list of tokens for text classification
    """
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', " ", text)  # Remove unusual symbols
    tokens = word_tokenize(text)
    words = [w for w in tokens if w not in stopwords.words("english")]  # Remove stopwords
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return (clean_tokens)


def build_model(best_parameters):
    pipeline_svm = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=best_parameters["vect__max_df"],
                                 ngram_range=best_parameters["vect__ngram_range"],
                                 max_features=best_parameters["vect__max_features"])),
        ('tfidf', TfidfTransformer(use_idf=best_parameters["tfidf__use_idf"])),
        ('svm', MultiOutputClassifier(SVC(kernel=best_parameters["svm__estimator__kernel"])))
    ])


def tune_model():
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


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)  # Fit the model
    y_pred = pd.DataFrame.from_records(y_pred)  # Store the predictions in a dataframe
    y_pred.columns = category_names

    # Creates lists to save the f1 score, recall and precision for each column
    f1scores = []
    recallscores = []
    precisionscores = []

    for col in X_test:
        print(col)
        f1score = f1_score(Y_test[col], y_pred[col], average="weighted")
        f1scores.append(f1score)
        print("F1 score: ", f1score)
        recallscore = recall_score(Y_test[col], y_pred[col], average="weighted")
        recallscores.append(recallscore)
        print("Recall score: ", recallscore)
        precisionscore = precision_score(Y_test[col], y_pred[col], average="weighted")
        precisionscores.append(precisionscore)
        print("Precision score: ", precisionscore)

        print("Mean F1 score svm model:", mean(f1scores))
        print("Mean recall svm simple model:", mean(recallscores))
        print("Mean precision score svm model:", mean(precisionscores))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Tuning model...")
        model = tune_model()
        model.fit(X_train, Y_train)
        best_param = model.best_params_
        print("Best parameters: ", best_param)

        print('Building model...')
        model = build_model(best_param)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()