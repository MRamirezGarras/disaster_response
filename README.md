# Disaster Response Pipeline Project

The objective of this project is to crate a Machine Learning algorithm able to classify twets in different 
categories related to natural disasters. This algorithm could greatly improve response times in case of 
an emergency.

## Files:
### data 
**disaster_categories.csv** stores the categories of each tweet
**disaster_messages** stores the text of each tweet
**process_data.py** merges the two previous files, cleans them and stores them in a SQL database
**DisasterResponse.db** is the database created using the previous file

### models 
**train_classifier** uses the SQL database to train a SVM algorithm to predict the category of a 
tweet based on its text
**svm_model_tuned.pkl** stores the model created with the previous file

### app 
**templates** contain the templates for the application
**run.py** runs the application. It shows visualizations of the training data and predicts categories of
texts that are input in a text box


## Running the code
This repository includes the SQL database and the SVM model, so it is not strictly necessary to run 
**process_data.py** or **train_classifier.py**. In case you want to run them, you can run the ETL 
pipeline with:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

And you can create the SVM model with:

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

To run the app you just go into the app directory and run:

`python run.py`

and then go to http://localhost:3001/
