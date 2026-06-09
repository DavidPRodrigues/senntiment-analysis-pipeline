import joblib

model = joblib.load('models/logistic_regression_model.pkl')
vectorizer =  joblib.load('models/tfidf_vectorizer.pkl')

while True:
    review =  input('enter a review or q to quit')

    if review.lower() == 'q':
        break

    review_vec = vectorizer.transform([review])

    prediction = model.predict(review_vec)[0]

    if prediction == 1:
        print('Positive review')
    else:
        print('Negative review')