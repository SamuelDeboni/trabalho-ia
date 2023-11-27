import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pydotplus
import scipy.stats as st
import xgboost as xgb
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
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
    precision_scorer_low_efficiency = make_scorer(precision_score, pos_label=0, average='binary')
    precision_scorer_high_efficiency = make_scorer(precision_score, pos_label=1, average='binary')

    recall_scorer_low_efficiency = make_scorer(recall_score, pos_label=0, average='binary')
    recall_scorer_high_efficiency = make_scorer(recall_score, pos_label=1, average='binary')

    f1_scorer_low_efficiency = make_scorer(f1_score, pos_label=0, average='binary')
    f1_scorer_high_efficiency = make_scorer(f1_score, pos_label=1, average='binary')

    precision_low_efficiency_std, precision_low_efficiency_mean, precision_ic_low_efficiency, precision_low_efficiency_values = \
        cross_val(model, x_train, y_train, precision_scorer_low_efficiency)
    precision_high_efficiency_std, precision_high_efficiency_mean, precision_ic_high_efficiency, precision_high_efficiency_values = \
        cross_val(model, x_train, y_train, precision_scorer_high_efficiency)

    recall_low_efficiency_std, recall_low_efficiency_mean, recall_ic_low_efficiency, recall_low_efficiency_values = \
        cross_val(model, x_train, y_train, recall_scorer_low_efficiency)
    recall_high_efficiency_std, recall_high_efficiency_mean, recall_ic_high_efficiency, recall_high_efficiency_values = \
        cross_val(model, x_train, y_train, recall_scorer_high_efficiency)

    f1_low_efficiency_std, f1_low_efficiency_mean, f1_ic_low_efficiency, f1_low_efficiency_values = \
        cross_val(model, x_train, y_train, f1_scorer_low_efficiency)
    f1_high_efficiency_std, f1_high_efficiency_mean, f1_ic_high_efficiency, f1_high_efficiency_values = \
        cross_val(model, x_train, y_train, f1_scorer_high_efficiency)

    metrics = {
        'precision_low_efficiency_values': precision_low_efficiency_values.tolist(),
        'precision_low_efficiency_std': precision_low_efficiency_std,
        'precision_low_efficiency_mean': precision_low_efficiency_mean,
        'precision_low_efficiency_ic': precision_ic_low_efficiency,
        'precision_high_efficiency_values': precision_high_efficiency_values.tolist(),
        'precision_high_efficiency_std': precision_high_efficiency_std,
        'precision_high_efficiency_mean': precision_high_efficiency_mean,
        'precision_high_efficiency_ic': precision_ic_high_efficiency,
        'recall_low_efficiency_values': recall_low_efficiency_values.tolist(),
        'recall_low_efficiency_std': recall_low_efficiency_std,
        'recall_low_efficiency_mean': recall_low_efficiency_mean,
        'recall_low_efficiency_ic': recall_ic_low_efficiency,
        'recall_high_efficiency_values': recall_high_efficiency_values.tolist(),
        'recall_high_efficiency_std': recall_high_efficiency_std,
        'recall_high_efficiency_mean': recall_high_efficiency_mean,
        'recall_high_efficiency_ic': recall_ic_high_efficiency,
        'f1_low_efficiency_values': f1_low_efficiency_values.tolist(),
        'f1_low_efficiency_std': f1_low_efficiency_std,
        'f1_low_efficiency_mean': f1_low_efficiency_mean,
        'f1_low_efficiency_ic': f1_ic_low_efficiency,
        'f1_high_efficiency_values': f1_high_efficiency_values.tolist(),
        'f1_high_efficiency_std': f1_high_efficiency_std,
        'f1_high_efficiency_mean': f1_high_efficiency_mean,
        'f1_high_efficiency_ic': f1_ic_high_efficiency
    }

    return metrics


def fit_and_evaluate(model, x_train, x_test, y_train, y_test, feature_names):
    model.fit(x_train, y_train)

    test_score = model.score(x_test, y_test)
    print(f"Test score {model.__class__.__name__}", test_score)
    metrics = compute_metrics(model, x_train, y_train)
    y_pred = model.predict(x_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # accuracy = accuracy_score(y_test, y_pred)

    class_report = ClassificationReport(model, classes=["Low Efficiency", "High Efficiency"])
    class_report.fit(x_train, y_train)
    class_report.score(x_test, y_test)
    class_report.show(outpath=f"{model.__class__.__name__}_class_report.png", clear_figure=True)

    prc = PrecisionRecallCurve(model, classes=["Low Efficiency", "High Efficiency"])
    prc.fit(x_train, y_train)
    prc.score(x_test, y_test)
    prc.show(outpath=f"{model.__class__.__name__}_precision_recall_curve.png", clear_figure=True)

    roc = ROCAUC(model, classes=["Low Efficiency", "High Efficiency"])
    roc.fit(x_train, y_train)
    roc.score(x_test, y_test)
    roc.show(outpath=f"{model.__class__.__name__}_roc.png", clear_figure=True)

    cm = ConfusionMatrix(model, classes=["Low Efficiency", "High Efficiency"])
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


def extract_rules_updated(tree, feature_names, total_samples):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, previous_rules):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            left_rules = previous_rules.copy()
            left_rules.append(f"{name} <= {round(threshold, 2)}")
            recurse(tree_.children_left[node], left_rules)

            right_rules = previous_rules.copy()
            right_rules.append(f"{name} > {round(threshold, 2)}")
            recurse(tree_.children_right[node], right_rules)
        else:
            predicted_class = tree_.value[node].argmax()
            if predicted_class is not None:
                rules.append({
                    "rule": " AND ".join(previous_rules),
                    "predicted_class": int(predicted_class),
                    "coverage": round(tree_.n_node_samples[node] / total_samples, 2),
                    "samples": int(tree_.n_node_samples[node])
                })

    recurse(0, [])

    rules = sorted(rules, key=lambda x: x['coverage'], reverse=True)

    return rules


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

        result = fit_and_evaluate(dtc_model, x_train, x_test, y_train, y_test, feature_names)

        tree_rules = extract_rules_updated(dtc_model, feature_names, 300)

        print(tree_rules)

        # Salva as regras em formato JSON
        with open("tree_rules.json", "w") as f:
            json.dump(tree_rules, f, indent=4)

        return result


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


def random_forest_grid_search():
    # open train and test sets
    with open('sleep_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        # Define the hyperparameter grid
        param_dist = {
            'n_estimators': [10, 50, 100, 150, 200, 250, 300],
            'max_features': ['log2', 'sqrt'],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': range(2, 10),
            'min_samples_split': range(2, 10),
        }

        # Perform grid search with 5-fold cross-validation
        grid_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=0),
            param_distributions=param_dist,
            cv=5,
            n_iter=100,
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)

        # Print the best hyperparameters and corresponding accuracy score
        print(f"Best parameters for RandomFlorest: {grid_search.best_params_}")
        print(f"Best accuracy score for RandomFlorest: {grid_search.best_score_}")

        rfc_model = RandomForestClassifier(**grid_search.best_params_, random_state=0)
        results = fit_and_evaluate(rfc_model, x_train, x_test, y_train, y_test, feature_names)

        importances = rfc_model.feature_importances_

        # Sort the features by importance in descending order
        indices = importances.argsort()[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for i in range(x_train.shape[1]):
            print("%d. feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))

        return results


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
        print("RandomForest Feature ranking:")
        for i in range(x_train.shape[1]):
            print("%d. feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))

        return results


def visualize_tree(model, feature_names):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=["Low Efficiency", "High Efficiency"],
        filled=True,
        rounded=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(f"{model.__class__.__name__}_graph.png")


def mlp_random_search():
    with open('sleep_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

    # Defina o espaço de parâmetros para a busca aleatória
    param_distributions = {
        'hidden_layer_sizes': [(10,), (5,), (2,), (3, 2), (4, 2), (5, 1), (5, 2), (5, 3, 1), (5, 2, 1)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
        'max_iter': [400, 500, 600, 700, 800],  # Increased max_iter range
        # Include momentum for sgd
        'momentum': [0.9, 0.95, 0.99] if 'solver' == 'sgd' else [0.0]  # Only for sgd solver
    }

    # Inicialize o MLPClassifier
    mlp = MLPClassifier()

    # Configurar o RandomizedSearchCV
    random_search = RandomizedSearchCV(
        mlp,
        param_distributions=param_distributions,
        n_iter=300,  # Número de iterações da busca aleatória
        cv=20,  # Número de folds na validação cruzada
        verbose=2,  # Para mensagens detalhadas
        random_state=42,
        n_jobs=-1  # Usa todos os núcleos da CPU
    )

    # Execute a busca com seus dados de treinamento (X_train, y_train)
    random_search.fit(x_train, y_train)

    # Mostre os melhores parâmetros e o desempenho do melhor modelo
    print("Melhores parâmetros:", random_search.best_params_)
    print("Melhor pontuação:", random_search.best_score_)


def mlp():
    with open('sleep_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        mlp_model = MLPClassifier(
            hidden_layer_sizes={5, 2, 1},
            max_iter=600,
            activation='tanh',
            alpha=0.001,
            learning_rate_init=0.01,
            solver='adam',
            random_state=42,
        )

    result = fit_and_evaluate(mlp_model, x_train, x_test, y_train, y_test, feature_names)

    results = permutation_importance(mlp_model, x_test, y_test, n_repeats=10, random_state=42)

    # Organizar os resultados e imprimir a importância das características
    print("MLP Feature ranking:")
    importance = results.importances_mean
    for i in range(x_train.shape[1]):
        print(f"Feature {feature_names[i]}: {importance[i]:.3}")

    return result


def xgboost_random_search():
    with open('sleep_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        param_dist = {
            'max_depth': [3, 4, 5, 6, 7, 9, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'n_estimators': [50, 100, 200, 400, 500, 600, 800],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Fração de colunas a serem usadas por árvore
            'min_child_weight': [1, 2, 3, 4, 5],  # Peso mínimo necessário para criar um novo nó na árvore
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],  # Parâmetro de poda da árvore
            'reg_alpha': [0, 0.1, 0.5, 1],  # Regularização L1
            'reg_lambda': [0, 0.1, 0.5, 1, 1.5, 2],  # Regularização L2
        }

        xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(y_train)), verbosity=2)

        grid_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            cv=20,
            n_iter=300,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(x_train, y_train)

        print("Best hyperparameters for XGBoost:", grid_search.best_params_)
        print("Best score for XGBoost:", grid_search.best_score_)

        # Avaliar o modelo com as melhores configurações
        best_xgb = xgb.XGBClassifier(**grid_search.best_params_)
        best_xgb.fit(x_train, y_train)

        return fit_and_evaluate(best_xgb, x_train, x_test, y_train, y_test, feature_names)


def xgboost():
    with open('sleep_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(set(y_train)),
            colsample_bytree=1.0,
            gamma=0.5,
            learning_rate=0.05,
            max_depth=10,
            min_child_weight=5,
            n_estimators=500,
            reg_alpha=0,
            reg_lambda=0,
            subsample=1
        )

        results = fit_and_evaluate(xgb_model, x_train, x_test, y_train, y_test, feature_names)

        importances = xgb_model.feature_importances_

        # Sort the features by importance in descending order
        indices = importances.argsort()[::-1]

        # Print the feature ranking
        print("XGBoost Feature ranking:")
        for i in range(x_train.shape[1]):
            print("%d. feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))

        return results


def create_graphic(tree_results, forest_results, neural_results, xgboost_results):
    # Substitua esses valores pelas suas métricas
    tree_results = json.loads(tree_results)
    forest_results = json.loads(forest_results)
    neural_results = json.loads(neural_results)
    xgboost_results = json.loads(xgboost_results)

    NN = [neural_results['precision'], neural_results['recall'], neural_results['f1_score']]  # NeuralNetwork
    RF = [forest_results['precision'], forest_results['recall'], forest_results['f1_score']]  # Random Forest
    DT = [tree_results['precision'], tree_results['recall'], tree_results['f1_score']]  # DecisionTree
    XGB = [xgboost_results['precision'], xgboost_results['recall'], xgboost_results['f1_score']]  # XGBoost

    # Crie uma lista com os nomes das métricas
    labels = ['Precision', 'Recall', 'F1 Score']

    # Configurando a posição das barras no eixo X
    barWidth = 0.2  # Ajuste a largura para acomodar a barra adicional
    r1 = np.arange(len(NN))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]  # Adicione uma nova posição para XGBoost

    # Criando as barras
    plt.figure(figsize=(10, 6))  # Ajuste o tamanho do gráfico conforme necessário
    bar1 = plt.bar(r1, NN, color='b', width=barWidth, edgecolor='grey', label='NeuralNetwork')
    bar2 = plt.bar(r2, RF, color='g', width=barWidth, edgecolor='grey', label='RandomForest')
    bar3 = plt.bar(r3, DT, color='r', width=barWidth, edgecolor='grey', label='DecisionTree')
    bar4 = plt.bar(r4, XGB, color='y', width=barWidth, edgecolor='grey',
                   label='XGBoost')  # Adicione a barra para XGBoost

    # Função para adicionar valor em cima da barra
    def add_values_on_bars(bars):
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), ha='center', va='bottom')

    # Adicionar valores nas barras
    add_values_on_bars(bar1)
    add_values_on_bars(bar2)
    add_values_on_bars(bar3)
    add_values_on_bars(bar4)  # Adicione valores para as barras XGBoost

    # Adicionando os nomes para o eixo X
    plt.xlabel('Métricas', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(NN))], labels)

    # Criando a legenda do gráfico
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4)  # Ajuste o número de colunas na legenda

    plt.savefig('graphic_results.png')
    # Exibindo o gráfico
    plt.show()


def compare_efficiency_metrics(results):
    # Estrutura para armazenar os pares de comparação e seus resultados
    comparison_results = {}

    # Convertendo strings JSON em dicionários
    parsed_results = [json.loads(result) for result in results]

    # Nomes dos algoritmos
    algorithms = [result['model_name'] for result in parsed_results]

    # Métricas para comparar
    metrics = ['precision', 'recall', 'f1']

    for metric in metrics:
        comparison_results[f"{metric}_high_efficiency"] = {}
        comparison_results[f"{metric}_low_efficiency"] = {}

        # Comparando cada algoritmo com os outros
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]

                # Comparação para alta eficiência
                alg1_high_metrics = parsed_results[i][f"{metric}_high_efficiency_values"]
                alg2_high_metrics = parsed_results[j][f"{metric}_high_efficiency_values"]
                t_stat_high, p_value_high = stats.ttest_ind(alg1_high_metrics, alg2_high_metrics)
                comparison_key_high = f"{alg1} vs {alg2} (High Efficiency)"
                comparison_results[f"{metric}_high_efficiency"][comparison_key_high] = {'t_statistic': t_stat_high,
                                                                                        'p_value': p_value_high}

                # Comparação para baixa eficiência
                alg1_low_metrics = parsed_results[i][f"{metric}_low_efficiency_values"]
                alg2_low_metrics = parsed_results[j][f"{metric}_low_efficiency_values"]
                t_stat_low, p_value_low = stats.ttest_ind(alg1_low_metrics, alg2_low_metrics)
                comparison_key_low = f"{alg1} vs {alg2} (Low Efficiency)"
                comparison_results[f"{metric}_low_efficiency"][comparison_key_low] = {'t_statistic': t_stat_low,
                                                                                      'p_value': p_value_low}

    return comparison_results


if __name__ == '__main__':
    # decision_tree_grid_search()
    # mlp_random_search()
    # xgboost_random_search()
    # random_forest_grid_search()

    mlp_results = mlp()
    tree_results = decision_tree()
    forest_results = random_forest()
    xgboost_results = xgboost()

    results = [tree_results, forest_results, xgboost_results, mlp_results]

    print("\n")

    comparison_results = compare_efficiency_metrics(results)

    combined_results = {
        'model_results': results,
        'comparison_results': comparison_results
    }

    # Exemplo de como acessar os resultados
    for metric in comparison_results:
        print(f"Comparação para {metric}:")
        for comparison in comparison_results[metric]:
            print(
                f"{comparison}: t-statistic = {comparison_results[metric][comparison]['t_statistic']}, p-value = {comparison_results[metric][comparison]['p_value']}")
        print("\n")

    create_graphic(tree_results, forest_results, mlp_results, xgboost_results)

with open('results.json', 'w') as f:
    # Use json.dump to write the results list to a file
    json.dump(combined_results, f, indent=4)
