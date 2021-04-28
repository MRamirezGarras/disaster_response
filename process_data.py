import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Read two datasets and merege them in one dataframe.
    :input messages data, categories data
    :output dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")

    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    #extract column names from the first row of categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:len(x) - 2])
    #Use the list of names as column names
    categories.columns = category_colnames
    #Convert the categories values to 0 or 1
    for col in categories.columns:
        categories[col] = categories[col].astype(str).str.split("-").str[1]
        categories[col] = pd.to_numeric(categories[col])
    # There are two problems
    # - related has a number 2
    # - child_alone has only elements of one category
    categories.related[categories.related == 2] = 1
    #Replace categories column with the new categories df
    categories.drop("child_alone", 1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    #drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    engine = create_engine("sqlite:///" + database_filename + ".db")
    df.to_sql(database_filename, engine, index=False)


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