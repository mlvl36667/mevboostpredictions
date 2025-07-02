import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import time

# Load data
dt = pd.read_csv('output/12s_auctions.csv')
data = dt[dt['maximal_bid'] > 0.04]

# Evaluation function
def eval_model(seconds, n_components=3, model_type='RFR'):
    columns = [
        'bid_count', 'unique_builder_pubkeys', 'average_value', 'std_deviation',
        'max_value', 'min_value', 'slope', 'intercept'
    ]
    selected_columns = [f"{col}_{sec}" for sec in seconds for col in columns]
    X = data[selected_columns]
    y = data['high_mev_block']

    X_clean = X.dropna()
    y_clean = y[X_clean.index]

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_clean)
    x_train, x_test, y_train, y_test = train_test_split(X_reduced, y_clean, test_size=0.2, random_state=42)

    start_time = time.time()

    if model_type == 'DTR':
        model = DecisionTreeRegressor(max_depth=10, random_state=42)
    elif model_type == 'RFR':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    elif model_type == 'KNN':
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_type == 'GBR':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42)
    elif model_type == 'SVR':
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    else:
        raise ValueError("Unknown model type")

    model.fit(x_train, y_train)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Model: {model_type} - Execution Time: {execution_time:.4f} seconds")

    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    precision = precision_score(y_test, y_pred_binary)
    return precision

# Parameters
time_intervals = range(2, 13)
components_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
results_dict = {}
colors = ['blue', 'green', 'orange', 'red', 'purple']
labels = ['DTR', 'RFR', 'KNN', 'GBR', 'SVR']

# Run models and collect results
for n_components in components_range:
    print(str(n_components))
    precisions_dtr = []
    precisions_rfr = []
    precisions_knn = []
    precisions_gbr = []
    precisions_svr = []

    for t in time_intervals:
        precisions_dtr.append(eval_model(list(range(t)), n_components=n_components, model_type='DTR'))
        precisions_rfr.append(eval_model(list(range(t)), n_components=n_components, model_type='RFR'))
        precisions_knn.append(eval_model(list(range(t)), n_components=n_components, model_type='KNN'))
        precisions_gbr.append(eval_model(list(range(t)), n_components=n_components, model_type='GBR'))
        precisions_svr.append(eval_model(list(range(t)), n_components=n_components, model_type='SVR'))

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(time_intervals, precisions_dtr, label='DTR', color=colors[0], marker='o')
    plt.plot(time_intervals, precisions_rfr, label='RFR', color=colors[1], marker='x')
    plt.plot(time_intervals, precisions_knn, label='KNN', color=colors[2], marker='^')
    plt.plot(time_intervals, precisions_gbr, label='GBR', color=colors[3], marker='d')
    plt.plot(time_intervals, precisions_svr, label='SVR', color=colors[4], marker='s')

#    plt.title(f'Precision for high_mev_block (SVD={n_components})', fontsize=18)
    plt.xlabel('t', fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'precision_high_mev_block_svd_{n_components}.pdf', format='pdf')
    plt.close()

