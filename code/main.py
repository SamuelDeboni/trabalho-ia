import pickle

import pandas as pd
import pydotplus
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold


def categorize_sleep_efficiency(efficiency):
    if efficiency < 0.7:
        return 'Low Efficiency'
    elif 0.7 <= efficiency < 0.85:
        return 'Medium Efficiency'
    else:
        return 'High Efficiency'


def extract_hour_and_minutes(data, column_name):
    bedtime = pd.to_datetime(data[column_name])
    return bedtime.dt.hour + bedtime.dt.minute / 60.0


def preprocessing():
    data = pd.read_csv("../Sleep_Efficiency.csv")
    # Remocao da Coluna ID
    data = data.drop('ID', axis=1)

    # Remocao de valores nulos
    data = data.dropna()

    # Converta as colunas de strings para datetime
    data['Bedtime'] = pd.to_datetime(data['Bedtime'])
    data['Wakeup time'] = pd.to_datetime(data['Wakeup time'])

    # Crie as características de hora e minuto para "Bedtime"
    data['Bedtime_hour_minute'] = extract_hour_and_minutes(data, "Bedtime")

    # Crie as características de hora e minuto para "Wakeup time"
    data['Wakeup_hour_minute'] = extract_hour_and_minutes(data, "Wakeup time")

    # Drop as colunas originais, se desejar
    data = data.drop(columns=['Bedtime', 'Wakeup time'])

    data['Sleep efficiency Category'] = data['Sleep efficiency'].apply(categorize_sleep_efficiency)

    le_gender = LabelEncoder()
    le_smoking_status = LabelEncoder()

    data["Gender"] = le_gender.fit_transform(data["Gender"])
    data["Smoking status"] = le_smoking_status.fit_transform(data["Smoking status"])

    # Caso haja duplicados, remova-os
    data = data.drop_duplicates()

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(columns=["Sleep efficiency", "Sleep efficiency Category"]),
        data["Sleep efficiency Category"],
        test_size=0.3,
    )

    print(x_train.columns.to_list())

    # Salvando os conjuntos de dados em um arquivo pickle
    with open("sleep_train_test.pkl", "wb") as f:
        pickle.dump((x_train.columns.to_list(), x_train.values, x_test.values, y_train.values, y_test.values), f)


def train_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model


def visualize_tree(model, feature_names):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=["Low Efficiency", "Medium Efficiency", "High Efficiency"],
        filled=True,
        rounded=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(f"{model.__class__.__name__}_graph.png")




if __name__ == '__main__':
    preprocessing()
    with open("sleep_train_test.pkl", "rb") as f:
        feature_names, x_train_data, x_test_data, y_train_data, y_test_data = pickle.load(f)

    tree_model = train_decision_tree(x_train_data, y_train_data)

    y_pred = tree_model.predict(x_test_data)

    print("Classification Report:")
    print(classification_report(y_test_data, y_pred))
    # evaluate_model(tree_model, x_test_data, y_test_data)


    visualize_tree(tree_model, feature_names)
