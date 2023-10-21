import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pydotplus
import scipy.stats as st
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from yellowbrick import ROCAUC
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, PrecisionRecallCurve


# Function for calculating confidence interval from cross-validation
def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values) - 1, loc=np.mean(values), scale=st.sem(values))


def cross_val(model, x_train, y_train, scorer):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, x_train, y_train, cv=kf, scoring=scorer)
    std = scores.std()
    mean_score = scores.mean()
    ic = interval_confidence(scores)
    return std, mean_score, ic, scores


def compute_metrics(model, x_train, y_train):
    precision_scorer_not_disease = make_scorer(precision_score, labels=[0], average='micro')
    precision_scorer_disease = make_scorer(precision_score, labels=[2], average='micro')

    recall_scorer_not_disease = make_scorer(recall_score, pos_label=0, average='micro')
    recall_scorer_disease = make_scorer(recall_score, pos_label=1, average='micro')

    f1_scorer_not_disease = make_scorer(f1_score, pos_label=0, average='micro')
    f1_scorer_disease = make_scorer(f1_score, pos_label=1, average='micro')

    precision_not_disease_std, precision_not_disease_mean, precision_ic_not_disease, precision_not_disease_values = \
        cross_val(model, x_train, y_train, precision_scorer_not_disease)
    precision_disease_std, precision_disease_mean, precision_ic_disease, precision_disease_values = \
        cross_val(model, x_train, y_train, precision_scorer_disease)

    recall_not_disease_std, recall_not_disease_mean, recall_ic_not_disease, recall_not_disease_values = \
        cross_val(model, x_train, y_train, recall_scorer_not_disease)
    recall_disease_std, recall_disease_mean, recall_ic_disease, recall_disease_values = \
        cross_val(model, x_train, y_train, recall_scorer_disease)

    f1_not_disease_std, f1_not_disease_mean, f1_ic_not_disease, f1_not_disease_values = \
        cross_val(model, x_train, y_train, f1_scorer_not_disease)
    f1_disease_std, f1_disease_mean, f1_ic_disease, f1_disease_values = \
        cross_val(model, x_train, y_train, f1_scorer_disease)

    metrics = {
        'precision_not_disease_values': precision_not_disease_values.tolist(),
        'precision_not_disease_std': precision_not_disease_std,
        'precision_not_disease_mean': precision_not_disease_mean,
        'precision_not_disease_ic': precision_ic_not_disease,
        'precision_disease_values': precision_disease_values.tolist(),
        'precision_disease_std': precision_disease_std,
        'precision_disease_mean': precision_disease_mean,
        'precision_disease_ic': precision_ic_disease,
        'recall_not_disease_values': recall_not_disease_values.tolist(),
        'recall_not_disease_std': recall_not_disease_std,
        'recall_not_disease_mean': recall_not_disease_mean,
        'recall_not_disease_ic': recall_ic_not_disease,
        'recall_disease_values': recall_disease_values.tolist(),
        'recall_disease_std': recall_disease_std,
        'recall_disease_mean': recall_disease_mean,
        'recall_disease_ic': recall_ic_disease,
        'f1_not_disease_values': f1_not_disease_values.tolist(),
        'f1_not_disease_std': f1_not_disease_std,
        'f1_not_disease_mean': f1_not_disease_mean,
        'f1_not_disease_ic': f1_ic_not_disease,
        'f1_disease_values': f1_disease_values.tolist(),
        'f1_disease_std': f1_disease_std,
        'f1_disease_mean': f1_disease_mean,
        'f1_disease_ic': f1_ic_disease
    }

    return metrics


def fit_and_evaluate(model, x_train, x_test, y_train, y_test, feature_names):
    model.fit(x_train, y_train)

    if isinstance(model, RegressorMixin):
        print("Oh yeah")

    test_score = model.score(x_test, y_test)
    print(f"Test score {model.__class__.__name__}", test_score)
    metrics = compute_metrics(model, x_train, y_train)
    y_pred = model.predict(x_test)

    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    # accuracy = accuracy_score(y_test, y_pred)

    class_report = ClassificationReport(model, classes=["Low Efficiency", "Medium Efficiency", "High Efficiency"])
    class_report.fit(x_train, y_train)
    class_report.score(x_test, y_test)
    class_report.show(outpath=f"{model.__class__.__name__}_class_report.png", clear_figure=True)

    prc = PrecisionRecallCurve(model, classes=["Low Efficiency", "Medium Efficiency", "High Efficiency"])
    prc.fit(x_train, y_train)
    prc.score(x_test, y_test)
    prc.show(outpath=f"{model.__class__.__name__}_precision_recall_curve.png", clear_figure=True)

    roc = ROCAUC(model, classes=["Low Efficiency", "Medium Efficiency", "High Efficiency"])
    roc.fit(x_train, y_train)
    roc.score(x_test, y_test)
    roc.show(outpath=f"{model.__class__.__name__}_roc.png", clear_figure=True)

    cm = ConfusionMatrix(model, classes=["Low Efficiency", "Medium Efficiency", "High Efficiency"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{model.__class__.__name__}_confusion_matrix.png")

    plt.close()

    if isinstance(model, DecisionTreeClassifier):
        visualize_tree(model, feature_names)

    results = {
        'model_name': model.__class__.__name__,
        'test_score': test_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    results.update(metrics)

    results_json = json.dumps(results)
    return results_json


def decision_tree_grid_search():
    # open train and test sets
    with open('sleep_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
        }

        # Perform a grid search with cross-validation
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=20),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        # Fit the grid search object to the training data
        grid_search.fit(x_train, y_train)

        # Print the best hyperparameters and the corresponding score
        print("Best hyperparameters for DecisionTree:", grid_search.best_params_)
        print("Best score for DecisionTree:", grid_search.best_score_)

        # Train a decision tree classifier on the training set
        dtc_model = DecisionTreeClassifier(**grid_search.best_params_, random_state=20)
        return fit_and_evaluate(dtc_model, x_train, x_test, y_train, y_test, feature_names)


def decision_tree():
    # open train and test sets
    with open('sleep_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)
        dtc_model = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=5,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=20
        )
        return fit_and_evaluate(dtc_model, x_train, x_test, y_train, y_test, feature_names)


def random_forest():
    # open train and test sets
    with open('sleep_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)
        rfc_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=2,
            min_samples_leaf=4,
            max_features='log2',
            criterion='entropy',
            random_state=20
        )

        results = fit_and_evaluate(rfc_model, x_train, x_test, y_train, y_test, feature_names)

        importances = rfc_model.feature_importances_

        # Sort the features by importance in descending order
        indices = importances.argsort()[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for i in range(x_train.shape[1]):
            print("%d. feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))

        return results


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
    decision_tree_grid_search()
# tree_results = decision_tree()
# forest_results = random_forest()
#
# results = [tree_results, forest_results]
#
# with open('results.json', 'w') as f:
#     # Use json.dump to write the results list to a file
#     json.dump(results, f, indent=4)
