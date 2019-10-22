# flask_pipeline
An interactive Flask web-application and data pipeline.  This example demonstrates analyzing tweets and categorizing those communications to aid in relief during a disaster.  The web-application can be used to deploy the ML model for use in the field.

# File Structure
Apps folder contains main module that runs data and ML pipelines and then launches a flask dashboard. Other folders of note include data which contains the datapipeline and processed database.  The model folder contains the ML model and the model checkpoint which can be loaded and used to classify incoming tweets.

# Detailed File Description:
1.  process_data.py: reads in the data, cleans and stores it in a SQL database. Basic usage is python process_data.py MESSAGES_DATA CATEGORIES_DATA NAME_FOR_DATABASE

2.  disaster_categories.csv and disaster_messages.csv (dataset)
DisasterResponse.db: created database from transformed and cleaned data.
Models

3.  train_classifier.py: includes the code necessary to load data, transform it using natural language processing, run a machine learning model using GridSearchCV and train it. Basic usage is python train_classifier.py DATABASE_DIRECTORY SAVENAME_FOR_MODEL

# Follow instructions below to launch the web-app:
1.  apps/run.py: Flask app and the user interface used to predict results and display them.
  a.  This shows an interactive dashboard where users can actively classify new messages

![website frontpage](https://github.com/BradEvanDavis/flask_pipeline/blob/master/Screenshot%20from%202019-10-21%2017-34-50.png)
