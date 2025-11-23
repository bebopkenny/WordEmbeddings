import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec # pip install gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Load a text dataset (documents) into a dataframe
docs_df = pd.read_csv('reviews.csv') # docs is a dataframe with columns 'id', 'text', and 'label'

# Preprocess text (tokenze + remove stopwords)
def preprocess(text):
    tokens = [t for t in simple_preprocess(text) if t not in STOPWORDS]
    return tokens

docs_df['tokens'] = docs_df['text'].apply(preprocess)

# Drop rows with empty tokens
tokenized_docs = [tokens for tokens in docs_df['tokens'] if len(tokens) > 0]
print("number of original documents:", len(docs_df))
print("number of non-empty documents:", len(tokenized_docs))
print("tokenized documents")
print(tokenized_docs)
#exit()

# Train Word2Vec on your tokens with sg=1 for skip-gram model (CBOW is default)
w2v_model = Word2Vec(sentences=tokenized_docs, sg=1, vector_size=20, window=5, min_count=1, workers=4)

def vectorize(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0) # instead of mean, other methods possible to combine vectors

# Vectorize the tokens
X = np.array([vectorize(tokens, w2v_model, 20) for tokens in docs_df['tokens']])
y = docs_df['label'].values
print("Feature matrix shape:", X)
print("Labels shape:", y)

# Train-test split and classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
