# Step 1: Import libraries
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load data
# df = pd.read_csv('data/Amazon_Reviews.csv')
df = pd.read_csv('data/Amazon_Reviews.csv', encoding='latin-1',engine='python', on_bad_lines='skip')

print(df.columns)
print(df.head())

# Step 3: Clean data
# df = df[['Text', 'Score']].dropna()
df = df[['Review Text', 'Rating']].dropna()
df.columns = ['Text', 'Score']

# Convert sentiment
# df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

# df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
# df = df.dropna(subset=['Score'])

# df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
# print(df['Score'].unique())

# Extract number from rating like "Rated 1 out of 5 stars"
df['Score'] = df['Score'].astype(str).str.extract('(\d+)')

# Convert to numeric
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# Drop invalid rows
df = df.dropna(subset=['Score'])

# Create sentiment
df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['clean_text'] = df['Text'].apply(clean_text)

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.2, random_state=42
)

# Step 5: Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))