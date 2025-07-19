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
import os

os.makedirs('ablation_results', exist_ok=True)

dt = pd.read_csv('output/12s_auctions.csv')
data = dt[dt['maximal_bid'] > 0.04]
print(str(len(data)))

def eval_model(seconds, n_components=3, model_type='RFR', target_variable="auction_length", ablated_features=None):
    columns = [
        'bid_count', 'unique_builder_pubkeys', 'average_value', 'std_deviation',
        'max_value', 'min_value', 'slope', 'intercept'
    ]
    
    if ablated_features:
        columns = [col for col in columns if col not in ablated_features]
    
    selected_columns = [f"{col}_{sec}" for sec in seconds for col in columns]
    X = data[selected_columns]
    y = data[target_variable]
    X_clean = X.dropna()
    y_clean = y[X_clean.index]
    svd = TruncatedSVD(n_components=min(n_components, len(selected_columns)), random_state=42)
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
    y_pred_rounded = np.round(y_pred).astype(int)
    mse = mean_squared_error(y_test, y_pred_rounded)
    mae = mean_absolute_error(y_test, y_pred_rounded)
    r2 = r2_score(y_test, y_pred_rounded)
    mape = np.mean(np.abs((y_test - y_pred_rounded) / y_test)) * 100
    return mse, mae, r2, mape

time_intervals = range(2, 13)
n_components = 4
target_variables = ['auction_length', 'bid_category']
target_variables = ['auction_length']
model_types = ['DTR', 'RFR', 'KNN', 'GBR', 'SVR']
features = ['bid_count', 'unique_builder_pubkeys', 'average_value', 'std_deviation', 'max_value', 'min_value', 'slope', 'intercept']

for target_variable in target_variables:
    print(f"Target variable: {target_variable}")
    
    baseline_results = {}
    for model_type in model_types:
        results = []
        for t in time_intervals:
            results.append(eval_model(list(range(t)), n_components=n_components, model_type=model_type, target_variable=target_variable))
        baseline_results[model_type] = results
    
    ablation_results = {}
    for feature in features:
        print(f"Ablating feature: {feature}")
        ablation_results[feature] = {}
        for model_type in model_types:
            results = []
            for t in time_intervals:
                results.append(eval_model(list(range(t)), n_components=n_components, model_type=model_type, target_variable=target_variable, ablated_features=[feature]))
            ablation_results[feature][model_type] = results
    
    metrics = ['MSE', 'MAE', 'R²', 'MAPE']
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    labels = ['DTR', 'RFR', 'KNN', 'GBR', 'SVR']
    
    for metric_idx, metric_name in enumerate(metrics):
        for page in range(2):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            plt.rcParams.update({'font.size': 14})
            
            start_idx = page * 4
            end_idx = min(start_idx + 4, len(features))
            
            for i, feature_idx in enumerate(range(start_idx, end_idx)):
                feature = features[feature_idx]
                ax = axes[i//2, i%2]
                
                for model_idx, model_type in enumerate(model_types):
                    baseline_metric = [result[metric_idx] for result in baseline_results[model_type]]
                    ablated_metric = [result[metric_idx] for result in ablation_results[feature][model_type]]
                    
                    ax.plot(time_intervals, baseline_metric, label=f'{model_type} (baseline)', color=colors[model_idx], linestyle='-', alpha=0.7)
                    ax.plot(time_intervals, ablated_metric, label=f'{model_type} (no {feature})', color=colors[model_idx], linestyle='--')
                
                ax.set_title(f'Ablation: {feature}', fontsize=16)
                ax.set_xlabel('t', fontsize=14)
                ax.set_ylabel(metric_name, fontsize=14)
                ax.legend(fontsize=10)
                ax.grid()
            
            for i in range(end_idx - start_idx, 4):
                axes[i//2, i%2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'ablation_results/ablation_{target_variable}_{metric_name.lower()}_page{page+1}.pdf', format='pdf')
            plt.close(fig)
    
    for page in range(2):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.rcParams.update({'font.size': 14})
        
        start_idx = page * 4
        end_idx = min(start_idx + 4, len(features))
        
        for i, feature_idx in enumerate(range(start_idx, end_idx)):
            feature = features[feature_idx]
            ax = axes[i//2, i%2]
            
            for model_idx, model_type in enumerate(model_types):
                baseline_r2 = [result[2] for result in baseline_results[model_type]]
                ablated_r2 = [result[2] for result in ablation_results[feature][model_type]]
                r2_diff = [ablated - baseline for baseline, ablated in zip(baseline_r2, ablated_r2)]
                
                ax.plot(time_intervals, r2_diff, label=f'{model_type}', color=colors[model_idx], marker='o')
            
            ax.set_title(f'{feature}', fontsize=16)
            ax.set_xlabel('t', fontsize=14)
            ax.set_ylabel('R² Change', fontsize=14)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.legend(fontsize=12)
            ax.grid()
        
        for i in range(end_idx - start_idx, 4):
            axes[i//2, i%2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'ablation_results/ablation_r2_change_{target_variable}_page{page+1}.pdf', format='pdf')
        plt.close(fig)
