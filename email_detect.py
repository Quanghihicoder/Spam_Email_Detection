import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def main(argv):
    if((len(argv) == 1) and (argv[0] == "LR" or argv[0] == "NB")):
        # loading the data from csv file to a pandas Dataframe
        raw_mail_data = pd.read_csv("./data/mail_data.csv")

        # replace the null values with a null string
        mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

        # label spam mail as 0;  ham mail as 1;
        mail_data.loc[mail_data['Category'] == 'spam', 'Category', ] = 0
        mail_data.loc[mail_data['Category'] == 'ham', 'Category', ] = 1

        # separating the data as texts and label
        X = mail_data['Message']
        Y = mail_data['Category']

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=3)

        # transform the text data to feature vectors that can be used as input to the model
        if (argv[0] == "LR"):
            feature_extraction = TfidfVectorizer(
                min_df=1, stop_words='english', lowercase='True')
        elif (argv[0] == "NB"):
            feature_extraction = CountVectorizer(
                min_df=1, stop_words='english', lowercase='True')

        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)

        # convert Y_train and Y_test values as integers
        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')

        if (argv[0] == "LR"):
            model = LogisticRegression()
            print("Using Logistic Regression")
        elif (argv[0] == "NB"):
            model = MultinomialNB()
            print("Using Naive Bayes")

        # training the model with the training data
        model.fit(X_train_features, Y_train)

        # prediction on training data
        prediction_on_training_data = model.predict(X_train_features)
        accuracy_on_training_data = accuracy_score(
            Y_train, prediction_on_training_data)
        print('Accuracy on training data : ', accuracy_on_training_data)

        # prediction on test data
        prediction_on_test_data = model.predict(X_test_features)
        accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
        print('Accuracy on test data : ', accuracy_on_test_data)

        print("Can you please some way remove all 'newlines' and make the message in just one line")
        email_message = input("Enter email message: ")
        input_mail = []
        input_mail.append(email_message)
        # convert text to feature vectors
        input_data_features = feature_extraction.transform(input_mail)

        # making prediction
        prediction = model.predict(input_data_features)

        print("\n")
        if (prediction[0] == 1):
            print('Result: Ham mail')

        else:
            print('Result: Spam mail')

    else:
        print("Wrong format")
        print("Correct format is: py email_detect.py <method_name>")
        print("Example: py email_detect.py NB")
        print("There are two methods available: LR - LogisticRegression; NB - Naive Bayes")


if __name__ == "__main__":
    main(sys.argv[1:])
