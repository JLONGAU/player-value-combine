

### Run this file in the Anaconda command prompt
# cd C:\\Users\\birch\\OneDrive\\Desktop\\Projects\\Geelong\\Player Value\\03. Notebooks
# python "250409 - AFL Combine Analysis_Model_1_Draft.py"

# Draft Prediction with Parallelization
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.storages import RDBStorage
from joblib import Parallel, delayed
from multiprocessing import Manager
import joblib


# Helper function for model evaluation
def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }

def train_model(model_name, model_info, pos_name):
    try:
        # Prepare data
        X = ml_data[columns_all]
        y = ml_data['Pct_Expected_Matches']  # Regression target

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        # Create Optuna study to minimize RMSE
        study = optuna.create_study(
                storage=f"sqlite:///{optuna_db_path}", direction="minimize" )
                    
        def objective(trial):
            params = {}
            for k, v in model_info["param_space"].items():
                if isinstance(v, list):
                    params[k] = trial.suggest_categorical(k, v)
                elif isinstance(v[0], float):
                    params[k] = trial.suggest_float(k, *v)
                elif isinstance(v[0], int):
                    params[k] = trial.suggest_int(k, *v)

            model = model_info["model"](**params)
            scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring="neg_root_mean_squared_error", n_jobs=1)
            return -np.mean(scores)  # Convert to positive RMSE

        study.optimize(objective, n_trials=50, timeout=7200)

        # Best params and model
        best_params = {k: v for k, v in study.best_trial.params.items() if k in model_info["param_space"]}
        print(f"Hyperparameters for {model_name}: {best_params}")

        model = model_info["model"](**best_params)
        model.fit(X_train, y_train)

        # Evaluate on test set
        metrics = evaluate_model(X_test, y_test, model)

        # Save best model (based on lowest RMSE)
        if metrics['rmse'] < best_model_dict["rmse"]:
            best_model_dict["rmse"] = metrics['rmse']
            best_model_dict["model"] = model
            best_model_dict["features"] = columns_all
            joblib.dump(model, f"04. Models/Draft/model_draft_predict_{pos_name}.pkl")

        print(f"{model_name} | R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.2f}")

        return {"model": model_name, **metrics}

    except Exception as e:
        print(f"Error in {model_name}: {e}")
        return None

if __name__ == '__main__':
              
    # Set configurations
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    np.random.seed(42)
    
    # File paths
    #file_path = 'C:\\Users\\birch\\OneDrive\\Desktop\\Projects\\Geelong\\Player Value'
    file_path =  r"C:\Users\john.long\player-value-combine"
    os.chdir(file_path)
    
    # Load dataset
    ml_data_full = pd.read_excel("02. Processed Data/AFL Combine Analysis_2018_2023.xlsx")
    
    # Identify all numerical columns dynamically (excluding categorical/text fields)
    numeric_columns = ml_data_full.select_dtypes(include=["number"]).columns.tolist()

    # Calculate overall percentage of non-NaN values for each numerical column
    overall_completeness = ml_data_full[numeric_columns].notna().mean() * 100

    min_completeness_threshold = 70  # Set threshold (e.g., 25% of players must have this test recorded)
    valid_tests = overall_completeness[overall_completeness > min_completeness_threshold].index.tolist()

    valid_tests = [test for test in valid_tests if test not in [
        'Unnamed: 0', 'Total_Matches', 'Draft_Success_Matches', 'Is_Draft_Success_Matches',
        "Is_Drafted", "Player_ID", "POS", 'POS_Gen Def', 'POS_Gen Fwd',
        'POS_Key Def', 'POS_Key Fwd', 'POS_Mid', 'POS_No matches', 'POS_Ruck',
        "Year", "Is_Father_Son", 'Pct_Expected_Matches']]

    #Define all columns for prediction
    columns_all = valid_tests

    #Save the column names for prediction
    columns_df = pd.DataFrame(columns_all, columns=['feature_names'])
    columns_df.to_csv('02. Processed Data/draft_model_feature_names.csv', index=False)

    # Models and hyperparameters
    models = {
        "Random Forest": {
            "model": RandomForestRegressor,
            "param_space": {
                'n_estimators': (50, 1000),
                'max_depth': (3, 50),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 20),
                'bootstrap': [True, False]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor,
            "param_space": {
                'n_estimators': (50, 1000),
                'max_depth': (3, 50),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 20),
                'learning_rate': (0.01, 0.3)
            }
        },
        "XGBoost": {
            "model": XGBRegressor,
            "param_space": {
                'n_estimators': (50, 1000),
                'max_depth': (3, 50),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0)
            }
        },
        "LightGBM": {
            "model": LGBMRegressor,
            "param_space": {
                'n_estimators': (50, 1000),
                'max_depth': (3, 100),
                'learning_rate': (0.001, 0.3),
                'num_leaves': (20, 300),
                'feature_fraction': (0.6, 1.0)
            }
        },
        "CatBoost": {
            "model": CatBoostRegressor,
            "param_space": {
                'iterations': (50, 1000),
                'depth': (3, 10),
                'learning_rate': (0.01, 0.3)
            }
        },
        "MLP Regressor": {
            "model": MLPRegressor,
            "param_space": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': (0.0001, 0.1),
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': (200, 2000)
            }
        }
    }
    
    positions = {
      #  "POS_Gen_Def": (ml_data_full['POS_Gen Def'] == 1) | (ml_data_full['POS_No matches'] == 1),
      #  "POS_Gen_Fwd": (ml_data_full['POS_Gen Fwd'] == 1) | (ml_data_full['POS_No matches'] == 1),
      #  "POS_Key_Def": (ml_data_full['POS_Key Def'] == 1) | (ml_data_full['POS_No matches'] == 1),
      #  "POS_Key_Fwd": (ml_data_full['POS_Key Fwd'] == 1) | (ml_data_full['POS_No matches'] == 1),
        "POS_Mid":     (ml_data_full['POS_Mid'] == 1)     | (ml_data_full['POS_No matches'] == 1),
        "POS_Ruck":    (ml_data_full['POS_Ruck'] == 1)    | (ml_data_full['POS_No matches'] == 1),
    }
    
    for pos_name, pos_filter in positions.items():
        print(f"\n=== Training models for {pos_name} ===")

        ml_data = ml_data_full[pos_filter].copy()
        ml_data = ml_data.dropna(subset=valid_tests + ['Pct_Expected_Matches'])

        # Save feature set for SHAP values
        ml_data[valid_tests].to_csv(f'02. Processed Data/draft_model_feature_df_{pos_name}.csv', index=False)

        manager = Manager()
        best_model_dict = manager.dict({"model": None, "rmse": np.inf, "features": []})

        optuna_db_path = f"optuna_study_{pos_name}.db"
        
        # Delete old Optuna DB
        if os.path.exists(optuna_db_path):
            os.remove(optuna_db_path)
        
        storage = RDBStorage(f"sqlite:///{optuna_db_path}")

    
        results = Parallel(n_jobs=max(1, int(os.cpu_count() * 0.7)))(
            delayed(train_model)(model_name, model_info, pos_name)
            for model_name, model_info in models.items()
        )

        results_df = pd.DataFrame([res for res in results if res])
        results_df.sort_values(by=['rmse'], ascending=True, inplace=True)
        results_df['Rank'] = results_df['rmse'].rank(ascending=True, method='dense')
        results_df.to_csv(f"05. Results/draft_model_results_{pos_name}.csv", index=False)
        
        


