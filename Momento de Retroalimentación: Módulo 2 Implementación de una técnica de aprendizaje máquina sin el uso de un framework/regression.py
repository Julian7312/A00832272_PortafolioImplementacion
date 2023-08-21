import pandas as pd
import matplotlib.pyplot as plt


def gd(train_X, train_y, a, i):

    b0 = 0
    b1 = 0 
    n = len(train_X)
    
    for _ in range(i):

        b0_gradient = 0
        b1_gradient = 0

        for i in range(n):

            x = train_X[i]
            y = train_y[i]

            b0_gradient += -(2/n) * (y - (b0 + b1 * x))
            b1_gradient += -(2/n) * x * (y - (b0 + b1 * x))
        
        b0 -= a * b0_gradient
        b1 -= a * b1_gradient
    
    return b0, b1

def splitData(df_class):
    split_ratio = 0.8
    split_index = int(len(df_class) * split_ratio)
    
    train_data = df_class.iloc[:split_index]
    test_data = df_class.iloc[split_index:]
    
    return train_data, test_data

columns = ["Class", "Alcohol", "Malic Acid", "Ash", "Alcalinity of ash", "Magnesium", "Total_phenols",
           "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color_intensity", "Hue", "Od", "Proline"]

df = pd.read_csv('/home/julian/Desktop/stats/wine.data', header=None, names=columns)

df_class_1 = df[df['Class'] == 1]
df_class_2 = df[df['Class'] == 2]
df_class_3 = df[df['Class'] == 3]


def graph(df):

    train_data, test_data = splitData(df)
    
    plt.scatter(train_data.Flavanoids, train_data.Alcohol, label='Training Data')
    plt.scatter(test_data.Flavanoids, test_data.Alcohol, label='Testing Data', marker='x')
    
    train_X = train_data.Flavanoids.values
    train_y = train_data.Alcohol.values
    test_X = test_data.Flavanoids.values
    
    a = 0.01
    i = 1000

    b0, b1 = gd(train_X, train_y, a, i)
    
    predictions = [b0 + b1 * x for x in test_X]
    
    plt.plot(test_X, predictions, color='red', label='Predicted Line')
    plt.xlabel('Flavanoids')
    plt.ylabel('Alcohol')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

graph(df_class_1)
graph(df_class_2)
graph(df_class_3)