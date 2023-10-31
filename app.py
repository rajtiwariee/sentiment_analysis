import streamlit as st
import string
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load stopwords explicitly
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Stopwords = set(stopwords.words("english"))


import pandas as pd

dataset = pd.read_csv('dataset.csv')


# Load the TF-IDF vectorizer
# with open('TF_IDF.pkl', 'rb') as file:
#     TF_IDF = pickle.load(file)

# import pickle
# Defining our vectorizer with total words of 5000 and with bigram model
TF_IDF = TfidfVectorizer(max_features = 5000, ngram_range = (2, 2))


TF_IDF.fit_transform(dataset["reviews"])

# Vectorize the preprocessed review
# vectorized_review = TF_IDF.transform([preprocessed_review])

# Load the model
filename = 'sentiment_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a function to preprocess and predict sentiment
def predict_sentiment(input_text):
    def Text_Cleaning(Text):
        Text = Text.lower()
        punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        Text = Text.translate(punc)
        Text = re.sub(r'\d+', '', Text)
        Text = re.sub('https?://\S+|www\.\S+', '', Text)
        Text = re.sub('\n', '', Text)
        return Text

    def Text_Processing(Text):
        Processed_Text = []
        Lemmatizer = WordNetLemmatizer()
        Tokens = nltk.word_tokenize(Text)
        for word in Tokens:
            if word not in Stopwords:
                Processed_Text.append(Lemmatizer.lemmatize(word))
        return " ".join(Processed_Text)

    cleaned_text = Text_Cleaning(input_text)
    processed_text = Text_Processing(cleaned_text)
    preprocessed_review = Text_Processing(Text_Cleaning(processed_text))
    return preprocessed_review

# Map sentiment to emojis
sentiment_emojis = {
    'negative': '😞',
    'neutral': '😐',
    'positive': '😄'
}

# Streamlit UI
st.title("Sentiment Analysis on Product Reviews")

# Set background to black and add background image
# Use forward slashes or escape the backslashes, and enclose the URL in double quotes
image_path = "C:/Users/HP SPECTRE/Desktop/sentiment_analysis/drum.jpg"
# Alternatively, you can use escaped backslashes as follows:
# image_path = "C:\\Users\\HP SPECTRE\\Desktop\\sentiment_analysis\\drum.jpg"

st.markdown(
    f"""
    <style>
    body {{
        background-image: url("{image_path}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        background-color: black;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_area("Enter the Product Review:")
if st.button("Analyze Sentiment"):
    if user_input:
        preprocessed_review = predict_sentiment(user_input)
        vectorized_review = TF_IDF.transform([preprocessed_review])
        predicted_class = loaded_model.predict(vectorized_review)
        sentimentS = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_sentiment = sentimentS[predicted_class[0]]
        sentiment_emoji = sentiment_emojis.get(predicted_sentiment, '😐')
        st.write(f"Predicted sentiment: {sentiment_emoji} {predicted_sentiment}")
    else:
        st.warning("Please enter a product review.")

st.text("This app performs sentiment analysis on product reviews.")
