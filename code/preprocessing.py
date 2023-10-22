import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def handle_missing_values(data):
    """
    Handle missing values in the dataset.

    Parameters:
    - data (DataFrame): Input data.

    Returns:
    - DataFrame: Data with imputed missing values.
    """

    # Impute missing values using KNN Imputer
    imputer = KNNImputer()
    df_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    return df_imputed


def inverse_categorize_sleep_efficiency(category):
    """
    Convert the category back to the corresponding sleep efficiency range.

    Parameters:
    - category (float): Category value (0.0, 0.5, or 1.0).

    Returns:
    - float: Midpoint value of the corresponding sleep efficiency range.
    """
    if category == 0.0:
        return 0  # Midpoint of the range [0, 0.7)
    elif category == 0.5:
        return 1  # Midpoint of the range [0.7, 0.85)
    else:  # category == 1.0
        return 2  # Midpoint of the range [0.85, 1]


def remove_outliers(data):
    """
    Detect and remove outliers using Isolation Forest.

    Parameters:
    - data (DataFrame): Input data.

    Returns:
    - DataFrame: Data without outliers.
    """
    isolation_forest = IsolationForest(contamination=0.05, random_state=20)
    outliers = isolation_forest.fit_predict(data)

    # Remove outliers from the original data
    cleaned_data = data[outliers == 1]

    return cleaned_data


def extract_hour_and_minutes(data, column_name):
    bedtime = pd.to_datetime(data[column_name])
    return bedtime.dt.hour + bedtime.dt.minute / 60.0


def categorize_sleep_efficiency(efficiency):
    """
    Categorize sleep efficiency based on the provided value.

    Parameters:
    - efficiency (float): A value representing sleep efficiency.

    Returns:
    - float: 0.0 for 'Low Efficiency', 0.5 for 'Medium Efficiency', and 1.0 for 'High Efficiency'.

    """
    if efficiency < 0.8:
        return 0  # Corresponds to 'Low Efficiency'
    else:
        return 1  # Corresponds to 'High Efficiency'


def feature_engineering(data):
    """
    Perform feature extraction and encoding.

    Parameters:
    - data (DataFrame): Input data.

    Returns:
    - DataFrame: Data with new features and encoded values.
    """
    # Remocao da Coluna ID
    data = data.drop('ID', axis=1)

    # Convert string columns to datetime
    data['Bedtime'] = pd.to_datetime(data['Bedtime'])
    data['Wakeup time'] = pd.to_datetime(data['Wakeup time'])

    # Extract hour and minute features for "Bedtime" and "Wakeup time"
    data['Bedtime_hour_minute'] = extract_hour_and_minutes(data, "Bedtime")
    data['Wakeup_hour_minute'] = extract_hour_and_minutes(data, "Wakeup time")

    # Drop original columns
    data.drop(columns=['Bedtime', 'Wakeup time'], inplace=True)

    # Encode "Gender" and "Smoking status" columns
    le_gender = LabelEncoder()
    le_smoking_status = LabelEncoder()
    data["Gender"] = le_gender.fit_transform(data["Gender"])
    data["Smoking status"] = le_smoking_status.fit_transform(data["Smoking status"])

    data['Sleep efficiency category'] = data['Sleep efficiency'].apply(categorize_sleep_efficiency)

    data = data.drop(columns=["Sleep efficiency"])

    return data


def feature_scaling(data, exclude_columns=None):
    """
    Apply Min-Max scaling to the dataset.

    Parameters:
    - data (DataFrame): Input data.
    - exclude_columns (list): List of column names to exclude from scaling.

    Returns:
    - DataFrame: Data after Min-Max scaling.
    """
    # Check if exclude_columns is provided
    if exclude_columns is None:
        exclude_columns = []

    # Separate the columns to be scaled and not to be scaled
    data_to_scale = data.drop(columns=exclude_columns)
    data_not_to_scale = data[exclude_columns]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_scale)

    # Convert scaled data back to DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns)

    # Merge the scaled and not scaled data back together
    result_df = pd.concat([scaled_df, data_not_to_scale], axis=1)


    return result_df


def check_balance(data):
    class_count = [0, 0, 0]
    total_count = 0

    for c in data['Sleep efficiency category']:
        class_count[int(c)] += 1
        total_count += 1

    print("Low     efficiency count is ", class_count[0])
    print("High  efficiency count is ", class_count[1])
    print("total_count is ", total_count)


def correlation(dataset):
    correlation_matrix = dataset.corr(method='pearson')
    print(correlation_matrix)

    plt.figure(figsize=(16, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()


def preprocessing():
    data = pd.read_csv("../Sleep_Efficiency.csv")
    print(f"Quantidade de instâncias antes do pré-processamento: {data.shape[0]}")

    # Caso haja duplicados, remova-os
    data = data.drop_duplicates()
    data = feature_engineering(data)
    data = handle_missing_values(data)
    check_balance(data)
    data = remove_outliers(data)
    print(f"Quantidade de instâncias apos remocao de outliers: {data.shape[0]}")
    check_balance(data)
    #data = feature_scaling(data, ["Sleep efficiency category"])

    # Remove features que possuem correlação com alguma outra feature
    data = data.drop(columns=["Light sleep percentage", "Wakeup_hour_minute", "Deep sleep percentage"])

    correlation(data.drop(columns=[]))

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(columns=["Sleep efficiency category"]),
        data["Sleep efficiency category"],
        test_size=0.3,
        random_state=20
    )

    print(f"Quantidade de instâncias após o pré-processamento: {data.shape[0]}")

    # Salvando os conjuntos de dados em um arquivo pickle
    with open("sleep_train_test.pkl", "wb") as f:
        pickle.dump((x_train.columns.to_list(), x_train.values, x_test.values, y_train.values, y_test.values), f)


if __name__ == '__main__':
    preprocessing()
