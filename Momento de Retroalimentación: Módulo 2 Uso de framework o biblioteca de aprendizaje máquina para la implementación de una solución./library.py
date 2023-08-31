import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt



columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

df = pd.read_csv('/home/julian/Desktop/stats/data/abalone.data', header=None, names=columns)

word_to_number = {
    'M':0,
    'F':1,
    'I':2
}

# cleaning data in Sex column
df['Sex'] = df['Sex'].map(word_to_number)

# assign features
feature1 = 'Diameter'
feature2 = 'Height'
feature3 = 'Shucked weight'

print(df)

def graph(df,feature):

    X = df[[feature]]
    y = df['Rings']
    
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #test metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    #graph results
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.title('Linear Regression')
    plt.show()

    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(comparison.head(10))

graph(df, feature1)
graph(df, feature2)
graph(df, feature3)
