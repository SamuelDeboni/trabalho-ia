import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree


def preprocessing():
    data = pd.read_csv("Sleep_Efficiency.csv")
    # Remocao da Coluna ID
    data = data.drop('ID', axis=1)

    # Remocao de valores nulos
    data = data.dropna()

    # Converta as colunas de strings para datetime
    data['Bedtime'] = pd.to_datetime(data['Bedtime'])
    data['Wakeup time'] = pd.to_datetime(data['Wakeup time'])

    # Crie as características de hora e minuto para "Bedtime"
    data['Bedtime_hour'] = data['Bedtime'].dt.hour
    data['Bedtime_minute'] = data['Bedtime'].dt.minute

    # Crie as características de hora e minuto para "Wakeup time"
    data['Wakeup_hour'] = data['Wakeup time'].dt.hour
    data['Wakeup_minute'] = data['Wakeup time'].dt.minute

    # Drop as colunas originais, se desejar
    data = data.drop(columns=['Bedtime', 'Wakeup time'])

    le_gender = LabelEncoder()
    le_smoking_status = LabelEncoder()

    data["Gender"] = le_gender.fit_transform(data["Gender"])
    data["Smoking status"] = le_smoking_status.fit_transform(data["Smoking status"])

    # Caso haja duplicados, remova-os
    data = data.drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(columns=["Sleep efficiency"]),
        data["Sleep efficiency"],
        test_size=0.3,
    )

    # Salvando os conjuntos de dados em um arquivo pickle
    with open("sleep_train_test.pkl", "wb") as f:
        pickle.dump((x_train.columns.to_list(), x_train.values, x_test.values, y_train.values, y_test.values), f)


def train_decision_tree(x_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Mean Squared Error: {mse}")
    return mse


def visualize_tree(model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=feature_names, rounded=True, fontsize=10)
    plt.show()


if __name__ == '__main__':
    preprocessing()
    with open("sleep_train_test.pkl", "rb") as f:
        feature_names, x_train_data, x_test_data, y_train_data, y_test_data = pickle.load(f)

    x_train = pd.DataFrame(x_train_data, columns=feature_names)
    x_test = pd.DataFrame(x_test_data, columns=feature_names)
    y_train = pd.Series(y_train_data)
    y_test = pd.Series(y_test_data)

    tree_model = train_decision_tree(x_train, y_train)
    evaluate_model(tree_model, x_test, y_test)
    visualize_tree(tree_model, feature_names)
