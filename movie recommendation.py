import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'movie_id': [101, 102, 103, 104, 105],
    'title': ['Dark World', 'Love in Paris', 'Cyberstorm', 'Heartbeats', 'Mega Heroes'],
    'genres': ['Sci-Fi Adventure', 'Romance Drama', 'Sci-Fi Action', 'Romance Musical', 'Action Superhero']
})

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genres'])

similarity_matrix = cosine_similarity(genre_matrix)

def recommend_movie(title):
    if title not in movies['title'].values:
        return f"❌ Movie '{title}' not found."

    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]

    recommended = [movies.iloc[i[0]]['title'] for i in sim_scores]
    return recommended

if __name__ == "__main__":
    movie = 'Dark World'
    print(f"\n🎬 Recommendations for '{movie}': {recommend_movie(movie)}")

    movie = 'Love in Paris'
    print(f"\n🎬 Recommendations for '{movie}': {recommend_movie(movie)}")
