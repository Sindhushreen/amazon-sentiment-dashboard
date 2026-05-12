# import streamlit as st
# import pickle

# # Load model
# model = pickle.load(open('model.pkl', 'rb'))
# vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# st.title("Customer Review Sentiment Analyzer")

# st.write("Enter a review to predict sentiment")

# user_input = st.text_area("Review")

# if st.button("Predict"):
#     if user_input:
#         input_vec = vectorizer.transform([user_input])
#         prediction = model.predict(input_vec)

#         if prediction[0] == 1:
#             st.success("Positive Review 😊")
#         else:
#             st.error("Negative Review 😠")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

st.title("Customer Review Insights Dashboard")

# Load data
df = pd.read_csv('data/Amazon_Reviews.csv', encoding='latin-1', engine='python', on_bad_lines='skip')

# Select needed columns
df = df[['Review Text', 'Rating']].dropna()
df.columns = ['Text', 'Score']

# Convert rating
df['Score'] = df['Score'].astype(str).str.extract('(\d+)')
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df = df.dropna(subset=['Score'])

# Sentiment
df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
min_rating = st.slider("Filter by minimum rating", 1, 5, 1)
df = df[df['Score'] >= min_rating]
# ---- SECTION 1: Overview ----
st.subheader("Dataset Overview")
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.write("Total Reviews:", len(df))
st.write("Positive Reviews:", len(df[df['sentiment']==1]))
st.write("Negative Reviews:", len(df[df['sentiment']==0]))

# ---- SECTION 2: Sentiment Chart ----
st.subheader("Sentiment Distribution")

fig, ax = plt.subplots()
df['sentiment'].value_counts().plot(kind='bar', ax=ax)
ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
st.pyplot(fig)

# ---- SECTION 3: Top Negative Words ----
st.subheader("Top Negative Keywords")

negative_reviews = df[df['sentiment'] == 0]['Text']

words = " ".join(negative_reviews).lower().split()
common_words = Counter(words).most_common(10)

for word, count in common_words:
    st.write(f"{word} : {count}")

# ---- SECTION 4: Sample Negative Reviews ----
st.subheader("Sample Negative Reviews")

for review in negative_reviews.head(5):
    st.write("- ", review)