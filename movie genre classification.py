import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def get_unique_genres(df, column_name):
    all_genres = set()
    for genres in df[column_name]:
        for genre in genres.split(','):
            all_genres.add(genre.strip())
    return all_genres

try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv('train_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading train_data: {e}")
    raise

unique_genres = get_unique_genres(train_data, 'GENRE')
genre_list = sorted(list(unique_genres)) 
fallback_genre = 'Unknown'

train_data['MOVIE_PLOT'] = train_data['MOVIE_PLOT'].astype(str).str.lower()
train_data['GENRE'] = train_data['GENRE'].str.split(',').apply(lambda x: [genre.strip() for genre in x])
mlb = MultiLabelBinarizer(classes=genre_list)
y_train = mlb.fit_transform(train_data['GENRE'])

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['MOVIE_PLOT'])

multi_output_classifier = MultiOutputClassifier(MultinomialNB())
multi_output_classifier.fit(X_train_tfidf, y_train)

try:
    with tqdm(total=50, desc="Loading Test Data") as pbar:
        test_data = pd.read_csv('test_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading test_data: {e}")
    raise

test_data['MOVIE_PLOT'] = test_data['MOVIE_PLOT'].astype(str).str.lower()

with tqdm(total=50, desc="Vectorizing Test Data") as pbar:
    X_test_tfidf = tfidf_vectorizer.transform(test_data['MOVIE_PLOT'])
    pbar.update(50)

with tqdm(total=50, desc="Predicting on Test Data") as pbar:
    y_pred = multi_output_classifier.predict(X_test_tfidf)
    pbar.update(50)

predicted_genres = mlb.inverse_transform(y_pred)
test_data['PREDICTED_GENRES'] = [genres if genres else [fallback_genre] for genres in predicted_genres]

with open("model_evaluation.txt", "w", encoding="utf-8") as output_file:
    for _, row in test_data.iterrows():
        movie_name = row['MOVIE_NAME']
        genre_str = ','.join(row['PREDICTED_GENRES'])
        output_file.write(f"{movie_name}:::{genre_str}\n")

y_train_pred = multi_output_classifier.predict(X_train_tfidf)
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred, average='micro', zero_division=0)
recall = recall_score(y_train, y_train_pred, average='micro', zero_division=0)
f1 = f1_score(y_train, y_train_pred, average='micro', zero_division=0)

with open("model_evaluation.txt", "a", encoding="utf-8") as output_file:
    output_file.write("\n\nModel Evaluation Metrics:\n")
    output_file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    output_file.write(f"Precision: {precision:.2f}\n")
    output_file.write(f"Recall: {recall:.2f}\n")
    output_file.write(f"F1-score: {f1:.2f}\n")

print("Model evaluation results and metrics have been saved to 'model_evaluation.txt'.")