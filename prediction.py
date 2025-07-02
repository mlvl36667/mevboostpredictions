import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.gridspec as gridspec
import time
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
dt = pd.read_csv('output/12s_auctions.csv')
#data = dt[dt['high_mev_block'] == 1]
data = dt[dt['maximal_bid'] > 0.04]
print(str(len(data)))
def eval_model(seconds, n_components=3, model_type='RFR', target_variable="auction_length"):
    columns = [
        'bid_count', 'unique_builder_pubkeys', 'average_value', 'std_deviation',
        'max_value', 'min_value', 'slope', 'intercept'
    ]
    selected_columns = [f"{col}_{sec}" for sec in seconds for col in columns]
    X = data[selected_columns]
    y = data[target_variable]
#    y = data['auction_length']
    X_clean = X.dropna()
    y_clean = y[X_clean.index]
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_clean)
    x_train, x_test, y_train, y_test = train_test_split(X_reduced, y_clean, test_size=0.2, random_state=42)
    start_time = time.time()  # Induló időpont

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
    y_pred_rounded = np.round(y_pred).astype(int)
    mse = mean_squared_error(y_test, y_pred_rounded)
    mae = mean_absolute_error(y_test, y_pred_rounded)
    r2 = r2_score(y_test, y_pred_rounded)
    mape = np.mean(np.abs((y_test - y_pred_rounded) / y_test)) * 100
    return mse, mae, r2, mape
time_intervals = range(2, 13)
components_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
results_dict = {}
target_variables = ['high_mev_block', 'auction_length', 'bid_category']
target_variables = ['auction_length', 'bid_category']
for target_variable in target_variables:
    print(target_variable)
    results_dict = {}
    for n_components in components_range:
        print(n_components)
        results_dtr = []
        results_rfr = []
        results_mlp = []
        results_gbr = []
        results_svr = []

        for t in time_intervals:
            results_dtr.append(eval_model(list(range(t)), n_components=n_components, model_type='DTR', target_variable=target_variable))
            results_rfr.append(eval_model(list(range(t)), n_components=n_components, model_type='RFR', target_variable=target_variable))
            results_mlp.append(eval_model(list(range(t)), n_components=n_components, model_type='KNN', target_variable=target_variable))
            results_gbr.append(eval_model(list(range(t)), n_components=n_components, model_type='GBR', target_variable=target_variable))
            results_svr.append(eval_model(list(range(t)), n_components=n_components, model_type='SVR', target_variable=target_variable))
        mses_dtr, maes_dtr, r2s_dtr, mapes_dtr = zip(*results_dtr)
        mses_rfr, maes_rfr, r2s_rfr, mapes_rfr = zip(*results_rfr)
        mses_mlp, maes_mlp, r2s_mlp, mapes_mlp = zip(*results_mlp)
        mses_gbr, maes_gbr, r2s_gbr, mapes_gbr = zip(*results_gbr)
        mses_svr, maes_svr, r2s_svr, mapes_svr = zip(*results_svr)
        if target_variable == 'high_mev_block':
         metrics = [
             ('MSE', mses_dtr, mses_rfr, mses_mlp, mses_gbr, mses_svr),
             ('MAE', maes_dtr, maes_rfr, maes_mlp, maes_gbr, maes_svr),
             ('R²', r2s_dtr, r2s_rfr, r2s_mlp, r2s_gbr, r2s_svr)
         ]
     
         # GridSpec elrendezés
         fig = plt.figure(figsize=(12, 10))
         spec = gridspec.GridSpec(2, 2, height_ratios=[1, 0.5], hspace=0.5)  # Magassági arány beállítása
     
         # Subplotok definiálása
         ax1 = fig.add_subplot(spec[0, 0])  # Bal felső
         ax2 = fig.add_subplot(spec[0, 1])  # Jobb felső
         ax3 = fig.add_subplot(spec[1, :])  # Alsó, középen
         axes = [ax1, ax2, ax3]
        else:
            metrics = [
                ('MSE', mses_dtr, mses_rfr, mses_mlp, mses_gbr, mses_svr),
                ('MAE', maes_dtr, maes_rfr, maes_mlp, maes_gbr, maes_svr),
                ('R²', r2s_dtr, r2s_rfr, r2s_mlp, r2s_gbr, r2s_svr),
                ('MAPE', mapes_dtr, mapes_rfr, mapes_mlp, mapes_gbr, mapes_svr)
            ]
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # 4 subplot 2x2 elrendezésben
        metrics = [('MSE', mses_dtr, mses_rfr, mses_mlp, mses_gbr, mses_svr),
                   ('MAE', maes_dtr, maes_rfr, maes_mlp, maes_gbr, maes_svr),
                   ('R²', r2s_dtr, r2s_rfr, r2s_mlp, r2s_gbr, r2s_svr)]
        if target_variable != 'high_mev_block':
         metrics.append(('MAPE', mapes_dtr, mapes_rfr, mapes_mlp, mapes_gbr, mapes_svr))
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        labels = ['DTR', 'RFR', 'KNN', 'GBR', 'SVR']
        plt.rcParams.update({'font.size': 22})
        for ax, (metric_name, dtr, rfr, mlp, gbr, svr) in zip(axes.flat, metrics):
            ax.plot(time_intervals, dtr, label=f'DTR', color=colors[0], marker='o')
            ax.plot(time_intervals, rfr, label=f'RFR', color=colors[1], marker='x')
            ax.plot(time_intervals, mlp, label=f'KNN', color=colors[2], marker='^')
            ax.plot(time_intervals, gbr, label=f'GBR', color=colors[3], marker='d')
            ax.plot(time_intervals, svr, label=f'SVR', color=colors[4], marker='s')
#            ax.set_title(f'', fontsize=18)
#            ax.set_title(f'{metric_name} (SVD={n_components}, Target={target_variable})', fontsize=18)
            ax.set_xlabel('t', fontsize=20)
            ax.set_ylabel(metric_name, fontsize=20)
            ax.tick_params(axis='x', labelsize=20)  # Increase x-tick label size
            ax.tick_params(axis='y', labelsize=20)  # Increase y-tick label size
            ax.legend(fontsize=18)
            ax.grid()
        plt.tight_layout(pad=3.0)
        plt.savefig(f'model_performance_{target_variable}_svd_{n_components}.pdf', format='pdf')
        plt.close(fig)


