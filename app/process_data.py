import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    '''loads messages and casts to type'''

    messages = pd.read_csv('messages.csv')
    messages['id'] = messages['id'].astype('int')
    categories = pd.read_csv('categories.csv')
    categories['id'] = categories['id'].astype('int')
    df = pd.merge(messages, categories, on='id', copy=False)
    return df


def clean_data(df):

    '''cleans data to be used in model'''

    categories = df['categories'].str.split(';', expand=True)
    categories = pd.concat([cats, df['id']], axis=1)
    categories.columns = cats.iloc[0, :]
    row = categories.columns
    category_colnames = [x[:-2] for x in row[:-1]]
    category_colnames.append('id')
    categories.columns = category_colnames

    for column in list(categories.columns[:-1]):
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str(x)[-1])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    df = df.drop('categories', axis=1)
    df = pd.merge(df, categories, on='id', copy=False)


def save_data(df, database_filename):
    """Save cleaned data into an SQLite database.

    Args:
    df: dataframe. Dataframe containing cleaned version of merged message and
    categories data.
    database_filename: string. Filename for output database.

    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterDatabase', engine, index=False, if_exists='replace')


def main():
    '''Executes pipeline'''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()