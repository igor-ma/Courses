from textblob import TextBlob


class SentimentModel:
    def __init__(self, phrase):
        tb = TextBlob(phrase)
        #tb = tb.translate(to='en')
        self.tb = tb

    def get_polarity(self):
        return self.tb.sentiment.polarity
        