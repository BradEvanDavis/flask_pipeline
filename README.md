# flask_pipeline
flask application and data pipeline example for analyzing tweets and categorizing those communications to aid in relief during a disaster.

# Processes to follow:
1.  process_data.py: reads in the data, cleans and stores it in a SQL database. Basic usage is python process_data.py MESSAGES_DATA CATEGORIES_DATA NAME_FOR_DATABASE

2.  disaster_categories.csv and disaster_messages.csv (dataset)
DisasterResponse.db: created database from transformed and cleaned data.
Models

3.  train_classifier.py: includes the code necessary to load data, transform it using natural language processing, run a machine learning model using GridSearchCV and train it. Basic usage is python train_classifier.py DATABASE_DIRECTORY SAVENAME_FOR_MODEL

# To run the app use the following syntax:
1.  run.py: Flask app and the user interface used to predict results and display them.

Example:
python run.py
