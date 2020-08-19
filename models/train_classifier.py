# import packages
import sys
import sqlite3
import joblib
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """
    Function to load csv file
    input: Data files, file path (csv)

    Output: pandas dataframe

    """
    # read in file
    cnx = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse_Table", cnx)
    print(df.head(2))


    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Function to preprocess the data
    inputs:
    Dataframe column with the massage

    Output: Tokenized column
    """

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenizing the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # remove not from stopwords
    all_stwords = stopwords.words('english')
    all_stwords.remove('not')
    words = [word for word in tokens if word not in all_stwords]


    # Reduce words to their stems
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]

    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in stemmed]

    return clean_tokens




def build_model():
    """
    Function to build and specify model parameters
    inputs: None

    Output: Model
    """

    # text processing and model pipeline
    pipeline = Pipeline([('cv3', CountVectorizer(tokenizer=tokenize)),
                        ('tdi', TfidfTransformer()),
                        ('nb_classifier', MultiOutputClassifier(BernoulliNB()))])


    # define parameters for GridSearchCV
    tuned_parameters = {'cv3__ngram_range': ((1, 1), (1, 2)),
                     'cv3__max_df': (0.5, 0.75, 1.0),
                     'cv3__max_features': (None, 5000, 10000),
                     'tdi__use_idf': (True, False)}



    # create gridsearch object and return as final model pipeline
    grid_search = GridSearchCV(estimator = pipeline, param_grid = tuned_parameters)


    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evalute the model performance
    inputs: Model, X_test and Y_test from the train_test_split, and y column unique names

    Output: None
    """

    Y_pred = model.predict(X_test)


    # Print out the full classification report
    results = pd.DataFrame(columns=['category', 'precision', 'recall', 'f_score'])
    count = 0
    for category in category_names:
        precision, recall, f_score, support = score(Y_test[category], Y_pred[:,count], average='weighted')
        results.at[count+1, 'category'] =category
        results.at[count+1, 'precision'] = precision
        results.at[count+1, 'recall'] = recall
        results.at[count+1, 'f_score'] = f_score
        count += 1
    avg_precision = results['precision'].mean()
    print(' %  Precision:', avg_precision)
    print(' %  Recall:', results['recall'].mean())
    print(' % f_score:', results['f_score'].mean())




def save_model(model, model_filepath):
    """
    Functoin to export the model to a pickle file
    inputs:
    model fit on the training dataset.
    File path to export the pickle file

    Output: None
    """
    # Export model as a pickle file
    #with open(model_filepath, 'wb') as f:
    #    pickle.dump(model, f)
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        print("\n Best Parameter :", model.best_params_)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
