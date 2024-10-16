import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Function to train the model
def train_model():
    # Load and preprocess the data
    raw_mail_data = pd.read_csv('mail_data.csv')
    mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

    # Strip any extra whitespace
    mail_data['Category'] = mail_data['Category'].str.strip()

    # Label spam as 0 and ham as 1
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

    # Ensure no NaNs remain
    mail_data = mail_data[mail_data['Category'].notnull()]

    # Print unique values to verify
    print("Unique categories:", mail_data['Category'].unique())

    # Separating data into features and labels
    X = mail_data['Message']
    Y = mail_data['Category'].astype(int)  # Ensure Y is an integer type

    # Print the data types to verify
    print("Y data type:", Y.dtypes)

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Feature extraction
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    # Save the model and vectorizer
    joblib.dump(model, 'model.pkl')
    joblib.dump(feature_extraction, 'vectorizer.pkl')

    print('Model and vectorizer saved!')

# Prediction system
def predict_mail(input_mail):
    # Load model and vectorizer
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Transform input data
    input_data_features = vectorizer.transform([input_mail])

    # Make prediction
    prediction = model.predict(input_data_features)

    return 'Ham mail' if prediction[0] == 1 else 'Spam mail'

if __name__ == '__main__':
    # Train the model if needed
    train_model()

    # Test with an example email
    example_email = "I've been searching for the right words to thank you for this breather"
    print(predict_mail(example_email))