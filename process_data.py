# import libraries
import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv("messages.csv")
messages.head()
# load categories dataset
categories = pd.read_csv("categories.csv")
categories.head()
# merge datasets
df = messages.merge(categories, on = "id")
df.head()
# create a dataframe of the 36 individual category columns
categories = df.categories.str.split(";", expand= True)
categories.head()
# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything
# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda x:x[0:len(x) -2])

print(category_colnames)
# rename the columns of `categories`
categories.columns = category_colnames
categories.head()

for col in categories.columns:
    categories[col] = categories[col].astype(str).str.split("-").str[1]
    categories[col] = pd.to_numeric(categories[col])
categories.head()

#Fix related column
categories.related[categories.related == 2] = 1
categories.related.unique()

# drop the original categories column from `df`
df.drop("categories", 1, inplace = True)
df.head()

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis = 1)
df.head()

# check number of duplicates
df.duplicated().sum()
# drop duplicates
df.drop_duplicates(inplace= True)
# check number of duplicates
df.duplicated().sum()

engine = create_engine('sqlite:///database_messages.db')
df.to_sql('database_messages', engine, index=False)