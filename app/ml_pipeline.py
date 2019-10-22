import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sqlalchemy import create_engine
import joblib


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(engine):

'''loads processed data for use in ML pipeline'''

    engine = create_engine(engine)
    df = pd.read_sql_table(con=engine, table_name='InsertTableName')
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y


def tokenize(text):
    '''tokenizes text from loaded data'''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):

        '''IDs verbs as this helps with categorization'''

        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def new_model_pipeline():

    '''Defines SKLearn ML pipeline'''

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier(),n_jobs=56))])
    return pipeline


def fit_model(X, Y, grid_search=True):

    '''Fits models, grid search options True/False'''

    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    model = new_model_pipeline()
    if grid_search == True:
        parameters = {
            'clf__estimator__base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3),
                                               DecisionTreeClassifier(max_depth=5)],
            'clf__estimator__n_estimators': [50, 100],
            'clf__estimator__learning_rate': [1., .75, .5]}
        cv = GridSearchCV(model, param_grid=parameters)
        grid = cv.fit(X_train, y_train)
        model = grid.best_estimator_.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test))
    else:
        model=model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    return model, y_pred


def predict(model):
    y_pred = pd.DataFrame(model.predict(X_test))
    return y_pred


def print_results(y_test, y_pred):

    '''Shows results from pipeline'''

    y_test = pd.DataFrame(y_test)
    labels = y_test.columns
    for x in range(len(y_test.columns)):
        print(labels[x])
        print(classification_report(y_test.iloc[:, x], y_pred.iloc[:, x]))
        print('\n')


def save_model(model):
    joblib.dump(model, 'saved_model.pkl')



