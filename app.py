import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punk')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained model and vectorizer
model = pickle.load(open('model_nlp.sav', 'rb'))
vector = pickle.load(open('vector_nlp.sav', 'rb'))

# Title of the Streamlit app
st.title("Disaster or Not?")

# Get user input
user_input = st.text_area("Enter a sentence for prediction:")


# Preprocess the user input
def preprocess_text(text):
    # Remove special characters
    text = re.sub('[^a-zA-Z0-9\s]', '', text)

    # Convert text to lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Remove words with less than 3 letters
    filtered_tokens = [word for word in filtered_tokens if len(word) >= 3]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in filtered_tokens]

    # Join tokens back into a single string
    return ' '.join(lemmatized_tokens)


# Button to trigger prediction
if st.button('Predict'):
    if user_input:
        # Preprocess the user input
        processed_text = preprocess_text(user_input)

        # Vectorize the preprocessed text (put it in a list)
        vectorized_input = vector.transform([processed_text])

        # Make the prediction
        prediction = model.predict(vectorized_input)

        # Display the result
        if prediction == 0:
            st.write('### Prediction: Not a disaster')
        else:
            st.write('### Prediction: Disaster')
    else:
        st.warning("Please enter a sentence to make a prediction.")