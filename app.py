from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
import pandas as pd

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['BIGDATA-PROJECT']
collection = db['MOVIES1']

# Load data from MongoDB into a DataFrame
cursor = collection.find()
data = pd.DataFrame(list(cursor))
client.close()

# Handling missing values in selected columns
selected_columns = ['Name of movie', 'Director', 'Star']
data[selected_columns] = data[selected_columns].fillna('')

# Convert non-string columns to string before combining
data['combined_features'] = data[selected_columns].astype(str).apply(lambda x: ' '.join(x), axis=1)

# Convert 'Rating_Category' column to string before applying LabelEncoder
data['Rating_Category'] = data['Rating_Category'].astype(str)
label_encoder = LabelEncoder()
data['Rating_Category'] = label_encoder.fit_transform(data['Rating_Category'])

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Define the target column
target = 'Rating_Category'

# Handling class imbalance using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
tfidf_matrix_res, target_res = oversampler.fit_resample(tfidf_matrix, data[target])

# Initialize and train a Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(tfidf_matrix_res, target_res)

# Get movie recommendations for a given movie title
def get_recommendations(movie_title, cosine_sim=linear_kernel(tfidf_matrix_res, tfidf_matrix_res), data=data):
    idx = data[data['Name of movie'].str.lower() == movie_title.lower()].index
    if not idx.empty:
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10 similar movies
        movie_indices = [i[0] for i in sim_scores]

        # Exclude the input movie from recommendations
        input_movie_index = data[data['Name of movie'].str.lower() == movie_title.lower()].index[0]
        if input_movie_index in movie_indices:
            movie_indices.remove(input_movie_index)

        # Ensure movie indices are within the DataFrame's bounds
        valid_indices = [idx for idx in movie_indices if idx < len(data)]
        unique_recommendations = []
        seen_titles = set()  # Use a set to track seen movie titles
        
        for idx in valid_indices:
            movie_title = data['Name of movie'].iloc[idx]
            if movie_title not in seen_titles:
                unique_recommendations.append(movie_title)
                seen_titles.add(movie_title)

        return unique_recommendations[:10]  # Limit recommendations to 10 unique movies
    else:
        return "Movie not found in the database."

# Flask endpoint to get movie recommendations
@app.route('/get_recommendations', methods=['POST'])
def recommend_movies():
    try:
        req_data = request.get_json()
        movie_title = req_data['searchText']
        recommendations = get_recommendations(movie_title)
        if isinstance(recommendations, list):
            return jsonify(recommendations)
        else:
            return jsonify({'message': recommendations})
    except KeyError:
        return jsonify({'error': 'Invalid JSON format. Missing "searchText" key.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
