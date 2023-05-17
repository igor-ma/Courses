import os
from logging import BASIC_FORMAT

from flask import Flask, jsonify, request
from flask_basicauth import BasicAuth
from mltools import MLStuff
from simple_model import SentimentModel

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
