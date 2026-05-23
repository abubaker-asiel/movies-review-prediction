import string
import pickle
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# ── Load dataset ───────────────────────────────────────────────────────────────
# Make sure IMDB Dataset.csv is in the same folder as this script
df = pd.read_csv('IMDB Dataset.csv')
print(f"Dataset loaded: {df.shape[0]} rows")

# ── Preprocessing ──────────────────────────────────────────────────────────────
def remove_punct(text):
    return "".join(c for c in text.lower() if c not in string.punctuation)

def tokenize(text):
    return word_tokenize(text)

def lemmatize(tokens):
    wn = nltk.WordNetLemmatizer()
    return [wn.lemmatize(w) for w in tokens]

print("Preprocessing... (this may take a few minutes)")
df['clean'] = df['review'].apply(remove_punct)
df['tokens'] = df['clean'].apply(tokenize)
df['lemmas'] = df['tokens'].apply(lemmatize)
df['text'] = df['lemmas'].apply(lambda x: ' '.join(x))

# ── Features & labels ──────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X = vectorizer.fit_transform(df['text'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['sentiment'])

# ── Train ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# ── Save pkl files ─────────────────────────────────────────────────────────────
pickle.dump(model, open('Review.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizier.pkl', 'wb'))
print("\n✅ Review.pkl and vectorizier.pkl saved successfully!")
