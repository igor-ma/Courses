from re import L, T
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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

