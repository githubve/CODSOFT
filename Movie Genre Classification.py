
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Genre List
genre_list = ['action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'family', 'fantasy', 'game-show', 'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller', 'war', 'western']

# Fallback Genre
fallback_genre = 'Unknown'

# Training Dataset
try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv("C:/Users/ADMIN/Downloads/archive (2)/Genre Classification Dataset/train_data.txt", sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading train_data: {e}")
    raise

train_data






train_data.describe(include='object').T





train_data.info()





train_data.duplicated().sum()





# Data Preprocessing for training Data
X_train = train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
genre_labels = [genre.split(', ') for genre in train_data['GENRE']]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(genre_labels)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)


with tqdm(total=50, desc="Vectorizing Train Data") as pbar:
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    pbar.update(50)










# Fit and transform the training data with progress bar
with tqdm(total=50, desc="Training Model") as pbar:
    naive_bayes = MultinomialNB()
    mul_op_classify = MultiOutputClassifier(naive_bayes)
    mul_op_classify.fit(X_train_tfidf, y_train)
    pbar.update(50)





# Load your test data
try:
    with tqdm(total=50, desc="Loading Test Data") as pbar:
        test_data = pd.read_csv("C:/Users/ADMIN/Downloads/archive (2)/Genre Classification Dataset/test_data.txt", sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading test_data: {e}")
    raise

test_data





test_data.describe(include='object').T





test_data.info() 





test_data.duplicated().sum()




# Data Preprocessing for test data
X_test = test_data["MOVIE_PLOT"].astype(str).apply(lambda doc: doc.lower())

# Transform the test data with progress bar
with tqdm(total=50, desc="Vectorizing Test Data") as pbar:
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    pbar.update(50)




# Predict genres on the test data
with tqdm(total=50, desc="Predicting on Test Data") as pbar:
    y_pred = mul_op_classify.predict(X_test_tfidf)
    pbar.update(50)

# Create a Data Frame for test with movie names and Predicted Genres
test_movie_names = test_data['MOVIE_NAME']
pred_gen = mlb.inverse_transform(y_pred)
test_results = pd.DataFrame({'MOVIE_NAME': test_movie_names, 'PREDICTED_GENRES': pred_gen})

# Replace empty unpredicted genres with the fallback genre
test_results['PREDICTED_GENRES'] = test_results['PREDICTED_GENRES'].apply(lambda genres: [fallback_genre] if len(genres) == 0 else genres)





# Write the results to an output text file with proper formatting
for _, row in test_results.iterrows():
    movie_name = row['MOVIE_NAME']
    genre_str = ', '.join(row['PREDICTED_GENRES'])
    print(f"{movie_name} ::: {genre_str}\n")











# Calculate evaluation metrics using training labels
y_train_pred = mul_op_classify.predict(X_train_tfidf)

# Calculate evaluation metrics
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred, average='micro')
recall = recall_score(y_train, y_train_pred, average='micro')
f1 = f1_score(y_train, y_train_pred, average='micro')

# Write the evaluation metrics
print("\n\nModel Evaluation Metrics:\n")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print(f"Precision: {precision:.2f}\n")
print(f"Recall: {recall:.2f}\n")
print(f"f1-score: {f1:.2f}\n")







