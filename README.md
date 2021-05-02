# Disaster Response Pipeline Project

Immediately after a natural disaster, emergency teams are usually saturated with messages asking
for help, reporting problems or just reporting the situation. Going through all these messages in 
a moment when a quick response is needed can hinder the emergency team´s answer. Having an automatic
method that can classify these messages would be a tremendous advantage, and it could help the
first responder to reach people in need much faster.

Twitter is on of the fastest channels used to communicate this kind of information. After a 
disaster, the number of tweets reporting the situation is huge. The objective of this project 
is to create a Machine Learning algorithm able to classify tweets in different categories related
to natural disasters. This could help to allocate each message to the appropiate receiver, and thus 
enhancing the response to the disaster. 

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

## Result

The model was created using a SVM algorithm with the text of 26215 tweets, which were classified using 35 different 
categories. Each tweet could belong to more than one category. The training step used 20%
of the tweets to generate the model. When is was tested in the testing set, the mean values of F1_score, 
recall and precission for all categories ranged from 0.93 to 0.95. 