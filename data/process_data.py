# import packages 

import sys
import pandas as pd
import numpy as np
import  sqlalchemy as db

def load_data(messages_filepath, categories_filepath):
    """
    Function to load csv file
    input: Data files, file path (csv)

    Output: pandas dataframe

    """
    # read in the csv file
    df_1 = pd.read_csv(messages_filepath)
    df_2 = pd.read_csv(categories_filepath)
    df = df_1.merge(df_2, left_on='id', right_on='id')
    return df

def clean_data(df):
    """
    Function to clean the dataframe
    inputs:
    Df :- Pandas dataframe

    Output: Cleaned pandas dataframe
    """

    # slipt the category column into 36 new columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, 0:]
    category_colnames = row.str.replace('-[0-9]', '', regex=True)

    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(-1)


    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # removing categories[related] when greater than 1
    categories = categories[categories.related != 2]

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # drop duplicates
    df.drop_duplicates(subset=['id'], keep='last', inplace=True)

    # dropping null values on the request column
    df = df[df['request'].notnull()]


    # reset index
    df.reset_index(drop=True, inplace=True)

    return df



def save_data(df, database_filename):
    """
    Func to save the cleaned df to an sqlite db
    inputs:
    Df :- Pandas dataframe
    Database_filename : Name to save the dataframe on the sqlite db as

    Output: None
    """
    # connect to sqlite
    engine = db.create_engine('sqlite:///{}'.format(database_filename))
    # drop table if it already exists
    engine.execute("DROP TABLE IF EXISTS DisasterResponse_Table")
    # save dataframe to the table
    df.to_sql('DisasterResponse_Table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print(df.head(2))

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
