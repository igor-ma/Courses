import os
from logging import BASIC_FORMAT

from flask import Flask, jsonify, request
from flask_basicauth import BasicAuth
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob


# Models
class MLStuff:
    def __init__(self, csv_path, target_column, test_size=0.3, 
        random_state=123):
        self.df = pd.read_csv(csv_path)
        self.test_size = test_size
        self.random_state = random_state
        self.model = LinearRegression()

        self.set_up_data(target_column)
        
    def set_up_data(self, target_column):
        X = self.df.drop(target_column, axis=1).values
        y = self.df[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def model_fit(self):
        self.model.fit(self.X_train, self.y_train)

    def model_predict(self, vars):
        return self.model.predict([vars])
    
class SentimentModel:
    def __init__(self, phrase):
        tb = TextBlob(phrase)
        #tb = tb.translate(to='en')
        self.tb = tb

    def get_polarity(self):
        return self.tb.sentiment.polarity
    

# Creating app
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

# Fixed code
mlt = MLStuff('../../data/processed/houses.csv', 'preco')
mlt.model_fit()

# Defining routes
@app.route('/')
def home():
    return 'Welcome to this simple API'

#test: http://34.95.247.142:5000/sentiment/I like this course
@app.route('/sentiment/<phrase>')
@basic_auth.required
def sentiment_polarity(phrase):
    sm = SentimentModel(phrase)
    return "Polarity for '{}': {}".format(
        phrase, sm.get_polarity())

#@app.route('/house/pricing/<int:size>')
#def house_price(size):
#    return mlt.model_predict(size)

@app.route('/house/pricing/', methods=['POST'])
@basic_auth.required
def house_price():
    data = request.get_json()
    input = [data[col] for col in data.keys()]
    price = mlt.model_predict(input)
    return jsonify(price=price[0])

# Running app
app.run(debug=True, host='0.0.0.0')
