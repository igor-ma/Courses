from logging import BASIC_FORMAT
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from simple_model import SentimentModel
from mltools import MLStuff

# Creating app
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = '123'

basic_auth = BasicAuth(app)

# Fixed code
mlt = MLStuff('houses.csv', 'preco')
mlt.model_fit()

# Defining routes
@app.route('/')
def home():
    return 'Welcome to this simple API'

@app.route('/sentiment/polarity/<phrase>')
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
app.run(debug=True)
