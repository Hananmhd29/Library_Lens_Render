from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from pymongo import MongoClient

# Load the preprocessed data
top_rated_books = pickle.load(open('top_rated_books.pkl', 'rb'))
all_books = pickle.load(open('all_books.pkl', 'rb'))

app = Flask(__name__)
app.secret_key = 'xyzsdfg'

# ✅ MongoDB connection
mongo_uri = os.environ.get("MONGO_URI")
client = MongoClient(mongo_uri)
db = client.get_database()   # gets database from URI
users_collection = db["user"]





#mysql = MySQL(app)

# Handle the presence or absence of 'Genre' column
if 'Genre' in all_books.columns:
    all_books['combined_features'] = all_books.apply(
        lambda x: x['Book'] + ' ' + x['Author'] + ' ' + x['Description'] + ' ' + x['Genre'], axis=1)
else:
    all_books['combined_features'] = all_books.apply(
        lambda x: x['Book'] + ' ' + x['Author'] + ' ' + x['Description'], axis=1)

# Create a TF-IDF vectorizer and transform the combined features into vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_books['combined_features'])


@app.route('/')
def index():
    if 'loggedin' in session:
        return render_template('index.html',
                               book_name=list(top_rated_books['Book'].values),
                               author=list(top_rated_books['Author'].values),
                               rating=list(top_rated_books['Avg_Rating'].values),
                               image=list(top_rated_books['Img_URL'].values),
                               description=list(top_rated_books['Description'].values))
    else:
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        library_id = request.form.get['library_id']
        password = request.form.get['password']

        user = users_collection.find_one({
            "library_id": library_id,
            "password": password
        })

            
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['library_id'] = user['library_id']
            message = 'Logged in successfully!'
            return render_template('index.html', message=message)
        else:
            message = 'Please enter correct library_id / password!'
    return render_template('loginnew.html', message=message)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('library_id', None)
    return redirect(url_for('login'))



@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST':
        userName = request.form.get['name']
        password = request.form.get['password']
        library_id = request.form.get['library_id']

        existing_user = users_collection.find_one({"library_id": library_id})
        
        if existing_user:
            message = 'Account already exists!'
        elif not userName or not password or not library_id:
            message = 'Please fill out the form!'
        else:
            users_collection.insert_one({
                "name": userName,
                "library_id": library_id,
                "password": password
            })
            message = 'You have successfully registered!'
           

    elif request.method == 'POST':
        message = 'Please fill out the form!'

    return render_template('registernew.html', message=message)
    
        


@app.route('/all_book')
def books_ui():
    return render_template('all_book.html',
                           book_name=list(all_books['Book'].values),
                           author=list(all_books['Author'].values),
                           rating=list(all_books['Avg_Rating'].values),
                           image=list(all_books['Img_URL'].values),
                           description=list(all_books['Description'].values))


@app.route('/recommend')
def recommend_ui():
     return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input').strip().lower()

    # Check if user input matches any book titles
    matches_title = all_books[all_books['Book'].str.lower() == user_input]

    # Check if user input matches any genres
    matches_genre = all_books[all_books['Genre'].str.lower() == user_input] if 'Genre' in all_books.columns else pd.DataFrame()

    # Check if user input matches any keywords
    matches_keyword = all_books[all_books['combined_features'].str.lower().str.contains(user_input, na=False)]

    if not matches_title.empty:
        # If user input matches book title, recommend books related to that book name and its genre
        idx = matches_title.index[0]
    elif not matches_genre.empty:
        # If user input matches genre, recommend books only from that genre
        similar_books = matches_genre
        return render_template('recommend.html',
                               book_name=list(similar_books['Book'].values),
                               author=list(similar_books['Author'].values),
                               rating=list(similar_books['Avg_Rating'].values),
                               image=list(similar_books['Img_URL'].values),
                               description=list(similar_books['Description'].values),
                               error_message=None)
    elif not matches_keyword.empty:
        # If user input matches keyword, recommend books only from that keyword
        similar_books = matches_keyword
        return render_template('recommend.html',
                               book_name=list(similar_books['Book'].values),
                               author=list(similar_books['Author'].values),
                               rating=list(similar_books['Avg_Rating'].values),
                               image=list(similar_books['Img_URL'].values),
                               description=list(similar_books['Description'].values),
                               error_message=None)
    else:
        # If user input matches neither book title, genre, nor keyword, provide error message
        return render_template('recommend.html', error_message='No books found matching the query. Please try another keyword or book.')

    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[:-11:-1]
    similar_books = all_books.iloc[similar_indices]

    # Sort similar books by average rating in descending order
    similar_books = similar_books.sort_values(by='Avg_Rating', ascending=False)

    return render_template('recommend.html',
                           book_name=list(similar_books['Book'].values),
                           author=list(similar_books['Author'].values),
                           rating=list(similar_books['Avg_Rating'].values),
                           image=list(similar_books['Img_URL'].values),
                           description=list(similar_books['Description'].values),
                           error_message=None)

if __name__ == '__main__':
    app.run(debug=True)
