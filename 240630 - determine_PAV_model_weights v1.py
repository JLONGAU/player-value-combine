# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:24:51 2024

@author: Birch Matthew
"""

### Determine PAV model weights

#%%

### Join B&F data onto Champion data

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSV
bf_data = pd.read_excel("data/B&F_2012_2023.xlsx")
stats_df = pd.read_csv("data/AFL Player Stats by Round by Season_Joined.csv")

### Plot the vote distributions
# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a figure to hold the subplots
plt.figure(figsize=(14, 10))

# Find unique seasons to plot
seasons = bf_data['Season'].unique()
seasons.sort()

# Create a subplot for each season
for i, season in enumerate(seasons, 1):
    plt.subplot(len(seasons) // 2 + 1, 2, i)
    sns.histplot(bf_data[bf_data['Season'] == season]['Votes'].dropna(), bins=15, kde=True, color='skyblue')
    plt.title(f'Vote Distribution in {season}')
    plt.xlabel('Votes')
    plt.ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Group the data by 'Season' and calculate the maximum votes for each season
max_votes_per_season = bf_data.groupby('Season')['Votes'].max()

# Apply the scaling transformation
bf_data['Scaled Votes'] = bf_data.apply(
    lambda row: (row['Votes'] / max_votes_per_season[row['Season']]) * 10,
    axis=1
)

# Strip leading and trailing whitespaces from all string columns
bf_data['Round'] = bf_data['Round'].str.strip()
bf_data['Opposition'] = bf_data['Opposition'].str.strip()

# Split name into its components
split_df = bf_data['Name'].str.split('\.| ', expand=True)
n_columns = 3
split_df = split_df.iloc[:, :n_columns]  
split_df['player_first_initial'] = split_df[0].str.strip().str[0]
split_df['player_last_name'] = split_df[1].str.cat(split_df[2], sep=' ', na_rep='').str.strip()
bf_data['player_first_initial'] = split_df['player_first_initial']
bf_data['player_last_name'] = split_df['player_last_name']
bf_data['player_last_name_lower_alpha_only'] = bf_data['player_last_name'].str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)

#Join onto player match statistics

#Filter Champion data for Geelong players
stats_df = stats_df[stats_df['SQUAD'] == 'GEE']

columns_to_keep = ['player_name', 'SEASON', 'ROUND', 'player_first_initial', 'player_last_name_lower_alpha_only',
                    'Disposals', 'Effective Disposals', 'Effective Kicks', 'Effective Handballs', 
                    'Goals', 'Behinds', 'POS',
                    'Ruck Contests', 'Hit Outs', 'Hit Outs to Advantage', 
                    'Goal Assist', 'Score Assist', 
                    'Inside 50s', #'Free Kick Differential', 
                    "Rebound 50's", 'Marks', 'Clearances', 
                    'CB Clearance', 'Tackles', 'Contested Possession', 
                    'Player Pressure Acts', 'Player Pressure Points', 'Score Involvments', 
                    'Intercept Marks', 'Contested Marks', 'Spoils', 'Ground Ball Gets', 
                    'clangers', 'marks_inside_fifty', 'one_percenters']

bf_data = pd.merge(bf_data[['Name', 'player_first_initial', 'player_last_name_lower_alpha_only', 'Season', 'Week', 'Scaled Votes']],
                    stats_df[columns_to_keep],
                    left_on = ['player_first_initial', 'player_last_name_lower_alpha_only', 'Season', 'Week'],
                    right_on = ['player_first_initial', 'player_last_name_lower_alpha_only', 'SEASON', 'ROUND'],
                    how = 'left')

bf_data.drop(columns = ['player_name', 'SEASON', 'ROUND'], inplace = True)

#Standardise player positions
position_simplified = {
    'Gen Def': 'DEF',
    'Gen Fwd': 'FWD',
    'Mid': 'MID',
    'Key Def': 'DEF',
    'Ruck': 'RUCK',
    'Mid-Fwd': 'MID',
    'Key Fwd': 'FWD',
    'Wing': 'MID'
}

# Replace the values in the column according to the mapping
bf_data['POS_simplified'] = bf_data['POS'].replace(position_simplified)

#Check the rows by position
print(f"Rows by position: \n{bf_data['POS'].value_counts()}")

bf_data.to_csv('data/bf_data.csv')


#%%

#Use Random Forest feature importances to determine the most important stats for each position

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import optuna
import warnings
import pickle

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSV
bf_data = pd.read_csv("data/bf_data.csv")

#Drop rows with NA
bf_data = bf_data.drop(columns=['Unnamed: 0']).dropna()

#Specify the stats columns
columns_all = ['Disposals', 'Effective Disposals', 'Effective Kicks', 'Effective Handballs',
                'Goals', 'Behinds', 'Goal Assist', 'Score Assist', 'Score Involvments',
                'Inside 50s',  'marks_inside_fifty',        
                'Contested Possession', 'Ground Ball Gets',
                'Marks', 'Contested Marks', 
                'Clearances', 'CB Clearance', 
                'Player Pressure Acts', 'Player Pressure Points', 'Tackles', 'one_percenters',
                'Intercept Marks', 'Spoils', 'Rebound 50\'s',
                'Ruck Contests', 'Hit Outs', 'Hit Outs to Advantage', 
                'clangers' ]

# Function to train and evaluate the LGBM model with optimized hyperparameters
def train_evaluate_model(X_train, y_train, X_test, y_test, best_params):
    #model = LGBMRegressor(**best_params)
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, rmse, r2, model

# Objective function for Optuna
def objective(trial):
    """# Define the hyperparameters to be tuned
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 0
    }"""
    
    # Define the hyperparameters to be tuned
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 100),  # -1 means no limit
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        #'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'random_state': 0
    }


    #model = LGBMRegressor(**param)
    model = RandomForestRegressor(**param)
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    mean_score = -np.mean(scores)  # Convert negative MSE back to positive
    
    return mean_score

# Create an empty dataframe to store feature importances
feature_importances_df = pd.DataFrame()
metrics_df = pd.DataFrame(columns=['POS', 'R2', 'MAE', 'RMSE'])

# Assuming bf_data and columns_all are already defined
for i in bf_data['POS'].unique():
    bf_data_pos = bf_data[bf_data['POS'] == i]

    print(f"Running position: {i}")

    # Select features and target
    X = bf_data_pos[globals()['columns_all']]
    y = bf_data_pos['Scaled Votes']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # Ensure all inputs are numpy arrays and handle missing data
    X_train = X_train.fillna(0).values
    y_train = y_train.fillna(0).values
    X_test = X_test.fillna(0).values
    y_test = y_test.fillna(0).values

    # Create the study and optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Get the best hyperparameters
    best_params = study.best_params

    # Train and evaluate the LGBM model with the best hyperparameters
    try:
        mae, mse, rmse, r2, model = train_evaluate_model(X_train, y_train, X_test, y_test, best_params)
        print(f"Position: {i}, R²: {r2}, MAE: {mae}, RMSE: {rmse}")

        # Store feature importances
        importances = model.feature_importances_
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        })
        feature_importances['POS'] = i
        feature_importances_df = pd.concat([feature_importances_df, feature_importances], ignore_index=True)
        
        # Store evaluation metrics
        metrics = pd.DataFrame({
            'POS': [i],
            'R2': [r2],
            'MAE': [mae],
            'RMSE': [rmse]
        })
        metrics_df = pd.concat([metrics_df, metrics], ignore_index=True)
        #metrics_df['R2'].mean()
    except Exception as e:
        print(f"Error during training for position {i}: {e}")
        continue

feature_importances_pivot = feature_importances_df.pivot(index='Feature', columns='POS', values='Importance')

# Define the positions
positions = feature_importances_df['POS'].unique()

# Create a dictionary to store the feature lists for each position
position_features = {pos: [] for pos in positions}

# Populate the dictionary with features having importance greater than 0.02
for pos in positions:
    position_features[pos] = feature_importances_df[(feature_importances_df['POS'] == pos) & 
                                                    (feature_importances_df['Importance'] > 0.01)]['Feature'].tolist()

# Unpack the dictionary into individual lists
columns_Gen_Fwd = position_features['Gen Fwd']
columns_Gen_Def = position_features['Gen Def']
columns_Mid = position_features['Mid']
columns_Key_Def = position_features['Key Def']
columns_Ruck = position_features['Ruck']
columns_Mid_Fwd = position_features['Mid-Fwd']
columns_Key_Fwd = position_features['Key Fwd']
columns_Wing = position_features['Wing']

#Save to csv
feature_importances_pivot.to_csv('data/feature_importances.csv')

# Dictionary to hold all the lists
position_features = {
    'Gen_Fwd': columns_Gen_Fwd,
    'Gen_Def': columns_Gen_Def,
    'Mid': columns_Mid,
    'Key_Def': columns_Key_Def,
    'Ruck': columns_Ruck,
    'Mid_Fwd': columns_Mid_Fwd,
    'Key_Fwd': columns_Key_Fwd,
    'Wing': columns_Wing
}

# Save the dictionary to a pickle file
with open('data/position_features.pkl', 'wb') as file:
    pickle.dump(position_features, file)

print("Position features saved to position_features.pkl")



#%%

### Use Linear Regression models to calculate coefficients for the the most important features

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pickle
import pandas as pd
import warnings
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from mlflow.models import infer_signature

import yaml

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSV
bf_data = pd.read_csv("data/bf_data.csv")

#Drop rows with NA
bf_data = bf_data.drop(columns=['Unnamed: 0']).dropna()

# Load the dictionary from the pickle file
with open('data/position_features.pkl', 'rb') as file:
    position_features = pickle.load(file)

# Unpack the dictionary into individual lists
columns_Gen_Fwd = position_features['Gen_Fwd']
columns_Gen_Def = position_features['Gen_Def']
columns_Mid = position_features['Mid']
columns_Key_Def = position_features['Key_Def']
columns_Ruck = position_features['Ruck']
columns_Mid_Fwd = position_features['Mid_Fwd']
columns_Key_Fwd = position_features['Key_Fwd']
columns_Wing = position_features['Wing']



"""
bf_data = bf_data.drop(columns = ['Week', 'Name', 'player_first_initial', 'player_last_name_lower_alpha_only',
                                  'Disposals', 'Effective Disposals', 'Ruck Contests',
                                  'Player Pressure Points', 'Player Pressure Acts','Free Kick Differential',
                                  'Goal Assist', 'one_percenters',  
                            ], axis=1)
"""

"""
columns_RUCK = ['Season', 'Player Pressure Acts', 'Player Pressure Points', 'Clearances', 'CB Clearance', 'Contested Marks', 'Hit Outs', 'Hit Outs to Advantage']
columns_FWD  = ['Season', 'Player Pressure Acts', 'Player Pressure Points', 'Ground Ball Gets', 'Contested Marks', 'Effective Kicks', 'Effective Handballs', 'Inside 50s', 'Goals', 'Behinds', 'Score Assist']
columns_DEF  = ['Season', 'Player Pressure Acts', 'Player Pressure Points', 'Intercept Marks', 'Contested Marks', 'Spoils', 'Score Involvments', 'Effective Kicks', 'Effective Handballs', 'Rebound 50\'s']
columns_MID  = ['Season', 'Player Pressure Acts', 'Player Pressure Points', 'Contested Possession', 'Clearances', 'CB Clearance',  'Tackles', 'Score Involvments', 'Effective Kicks', 'Effective Handballs', 'Inside 50s']
"""
"""
columns_all = [ 'Disposals', 'Effective Disposals', 'Effective Kicks', 'Effective Handballs',
                   'Goals', 'Behinds', 'Goal Assist', 'Score Assist', 'Score Involvments',
                   'Inside 50s',  'marks_inside_fifty',        
                   'Contested Possession', 'Ground Ball Gets',
                   'Marks', 'Contested Marks', 
                   'Clearances', 'CB Clearance', 
                   'Player Pressure Acts', 'Player Pressure Points', 'Tackles', 'one_percenters',
                   'Intercept Marks', 'Spoils', 'Rebound 50\'s',
                   'Ruck Contests', 'Hit Outs', 'Hit Outs to Advantage', 
                   'clangers' ]
"""

#Function to plot the correlation matrix
def view_correlation(X, position):
    # Compute the correlation matrix for numeric features
    correlation_matrix = X.corr()
    
    # Plot the correlation matrix using a heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=.5)
    plt.title(f'Correlation Matrix of Variables - {position}')
    plt.show()
    
    # Create a DataFrame from the correlation matrix stack
    correlation_pairs = correlation_matrix.stack().reset_index()
    correlation_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
    
    # Remove self-correlation (correlation of variables with themselves)
    correlation_pairs = correlation_pairs[correlation_pairs['Feature1'] != correlation_pairs['Feature2']]
    
    # Sort by absolute correlation values, highest first
    correlation_pairs_sorted = correlation_pairs.copy()
    correlation_pairs_sorted['Abs Correlation'] = correlation_pairs_sorted['Correlation'].abs()
    correlation_pairs_sorted = correlation_pairs_sorted.sort_values(by='Abs Correlation', ascending=False).drop('Abs Correlation', axis=1)
    
    print(f"The most correlated variables: \n\n {correlation_pairs_sorted.head(10)}")
    return


#Define functions to calculate weighted metrics
def weighted_mean_absolute_error(y_true, y_pred, sample_weight):
    return np.sum(sample_weight * np.abs(y_true - y_pred)) / np.sum(sample_weight)

def weighted_mean_squared_error(y_true, y_pred, sample_weight):
    return np.sum(sample_weight * (y_true - y_pred) ** 2) / np.sum(sample_weight)

def weighted_r2_score(y_true, y_pred, sample_weight):
    y_true_mean = np.average(y_true, weights=sample_weight)
    total_sum_of_squares = np.sum(sample_weight * (y_true - y_true_mean) ** 2)
    residual_sum_of_squares = np.sum(sample_weight * (y_true - y_pred) ** 2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)

# Objective function for optimization
def objective_function(coef, X, y, weights):
    predictions = X @ coef
    residuals = y - predictions
    weighted_residuals = weights * residuals
    return np.sum(weighted_residuals ** 2)

#Create an empty dataframe
weights_df = pd.DataFrame(columns = ['Feature'])

#Iterate over each of the positions
for i in bf_data['POS'].unique():
    #print(i)
    #i = 'Gen Def'
    
    bf_data_pos = bf_data[bf_data['POS'] == i]
    i_updated = i.replace(' ', '_').replace('-', '_')
    print(f"Running position: {i}")
    
    feature_list = globals()[f'columns_{i_updated}'] + ['Season']
    
    # Select features and target
    X = bf_data_pos[feature_list]
    #X = bf_data_pos[globals()[f'columns_{i}']]    
    #X = bf_data_pos.drop(columns = ['POS', 'Scaled Votes'], axis=1)
    y = bf_data_pos['Scaled Votes']
    
    #View correlation matrix
    view_correlation(X, i)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    #Determine number of seasons in the data
    num_seasons = bf_data['Season'].nunique() + 2
    #num_seasons = 1
    
    #Ensuring that more importance is given on more recent years by applying linearly increasing weights
    current_year = max(bf_data['Season'])         
    weights = X_train['Season'].apply(lambda x: (x - (current_year - num_seasons)) / num_seasons)
    # Ensure weights are not less than 0.05 and not greater than 1
    #weights = np.clip(weights, 0.05, 1)  
    weights = np.clip(weights, 1, 1)  
    
    X_test_season = X_test['Season']
    X_train.drop(columns=['Season'], axis=1, inplace=True)
    X_test.drop(columns=['Season'], axis=1, inplace=True)
    
    # Initial coefficients (start with zeros)
    initial_coef = np.zeros(X_train.shape[1])
    
    # Constraints to ensure non-negative coefficients
    constraints = [{'type': 'ineq', 'fun': lambda x: x}]
    
    # Perform the optimization
    result = minimize(objective_function, initial_coef, args=(X_train, y_train, weights), constraints=constraints)
    
    # Get the optimized coefficients
    optimized_coef = result.x
    
    # Apply the coefficients to the test set
    y_pred = X_test @ optimized_coef
        
    
    """
    # Feature selection using Lasso
    #lasso = Lasso(alpha=0.0)
    #lasso.fit(X_train, y_train, sample_weight=weights)
    #selected_features = np.where(lasso.coef_ != 0)[0]
   
    #X_train_selected = X_train.iloc[:, selected_features]
    #X_test_selected = X_test.iloc[:, selected_features]
      
    # Define the Linear Regression model
    #linear_model = LinearRegression()
    
    # Create a new pipeline with the linear regression model
    #linear_pipeline = Pipeline(steps=[
        #('model', linear_model)
    #])
    
    # Fit the model on the training data with weights
    #linear_pipeline.fit(X_train_selected, y_train, model__sample_weight=weights)
    
    # Predicting on the test set
    #y_pred = linear_pipeline.predict(X_test_selected)
    """
    
    """
    # Computing weighted evaluation metrics
    sample_weight_test = X_test_season.apply(lambda x: (x - (current_year - num_seasons)) / num_seasons)
    sample_weight_test = np.clip(sample_weight_test, 0.05, 1)
    
    mae = weighted_mean_absolute_error(y_test, y_pred, sample_weight_test)
    mse = weighted_mean_squared_error(y_test, y_pred, sample_weight_test)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = weighted_r2_score(y_test, y_pred, sample_weight_test)
    mae, mse, rmse, r2
    """
    
    # Computing evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)
    mae, mse, rmse, r2
    
    
    """
    # Evaluate on the most recent year
    recent_year_mask = (X_test_season == current_year)
    y_test_recent = y_test[recent_year_mask]
    y_pred_recent = y_pred[recent_year_mask]
    
    # Computing evaluation metrics on the most recent year
    mae = mean_absolute_error(y_test_recent, y_pred_recent)
    mse = mean_squared_error(y_test_recent, y_pred_recent)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_test_recent, y_pred_recent)
    mae, mse, rmse, r2
    """
    
    print(f'mae {mae} \nmse {mse} \nrmse {rmse} \nr2 {r2}')
    
    # Create a DataFrame to view feature names with their corresponding coefficients
    selected_feature_names = X_train.columns
    
    for feature, coef in zip(selected_feature_names, optimized_coef):
        if feature not in weights_df['Feature'].values:
            new_row = pd.DataFrame({'Feature': [feature]})
            weights_df = pd.concat([weights_df, new_row], ignore_index=True)
        weights_df.loc[weights_df['Feature'] == feature, f'{i}_Coefficient'] = coef * 5
        
            
    #Plot 1: Plot predicted v actuals
    plt.figure(figsize=(12, 6))
    plt.hist(y_test, bins=30, alpha=0.5, label='Actual Values', color='gray')
    plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted Values', color='orange')
    plt.title('Distribution of Actual vs. Predicted Values')
    plt.xlabel('Scaled Votes')
    plt.ylabel('Frequency')
    plt.legend()
    plot_path_1 = f'distribution_actual_vs_predicted_{i}.png'
    plt.savefig(plot_path_1)
    plt.show()
    plt.close()
    
    #Plot 2: Residuals plot
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plot_path_2 = f'residual_plot_{i}.png'
    plt.savefig(plot_path_2)
    plt.show()
    plt.close()
    
    #Plot 3: Prediction error plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_test, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Diagonal line
    plt.title('Prediction Error Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plot_path_3 = f'prediction_error_plot_{i}.png'
    plt.savefig(plot_path_3)
    plt.show()
    plt.close()
    
    """
    #Log the result in ML Flow
    # Create a new MLflow Experiment
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("PAV Weight Model")
    
    # Start an MLflow run
    with mlflow.start_run(run_name=f"Model for POS {i}"):
        
        # Log model info
        mlflow.log_params({
            "model_type": "Linear Regression",
            "position": i
        })

        # Log individual metrics
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        mlflow.log_metric("R2 Value", r2)

        # Get the schemas of the model inputs and outputs
        signature = infer_signature(X_train, linear_pipeline.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=linear_pipeline,
            artifact_path="weight_model",
            signature=signature,
            input_example=X_train,
            registered_model_name=f"Linear_Model_POS_{i}"
        )
        
        # Create tags
        mlflow.set_tag("model_type", "Linear Regression")
        mlflow.set_tag("description", f"Predict the weights for position {i}")
        
        #Save the plots
        mlflow.log_artifact(plot_path_1)
        mlflow.log_artifact(plot_path_2)
        mlflow.log_artifact(plot_path_3)
      """
      
# Display the DataFrame sorted by absolute values of coefficients for better visibility
print(f'Model weights: \n {weights_df}')    
weights_df.to_csv('data/weights_df_6.csv')

# Transform into YAML format
def convert_csv_to_yml(weights_df):
    # Define the positions
    positions = ['Gen Def', 'Gen Fwd', 'Mid', 'Key Def', 'Ruck', 'Mid-Fwd', 'Key Fwd', 'Wing']
    
    # Initialize the weights dictionary
    weights_dict = {f'{pos}_Weights': {} for pos in positions}

    # Iterate through each row in the dataframe
    for _, row in weights_df.iterrows():
        feature = row['Feature']
        for pos in positions:
            coefficient = row[f'{pos}_Coefficient']
            if pd.notna(coefficient):
                weights_dict[f'{pos}_Weights'][feature] = coefficient

    # Convert the dictionary to a YML formatted string
    weights_yml = yaml.dump(weights_dict, sort_keys=False)
    
    return weights_yml

# Assuming weights_df is the DataFrame loaded from the CSV
weights_yml = convert_csv_to_yml(weights_df)

# Save the YML content to a new file
output_yml_file_path = 'data/model_weights_v6.yml'
with open(output_yml_file_path, 'w') as file:
    file.write(weights_yml)

print(weights_yml)















#%%


#One random forest for all postiions 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import optuna


#Import CSV
bf_data = pd.read_csv("data/bf_data.csv")

#Drop rows with NA
bf_data = bf_data.drop(columns=['Unnamed: 0']).dropna()

columns_all = [ 'Disposals', 'Effective Disposals', 'Effective Kicks', 'Effective Handballs',
                   'Goals', 'Behinds', 'Goal Assist', 'Score Assist', 'Score Involvments',
                   'Inside 50s',  'marks_inside_fifty',        
                   'Contested Possession', 'Ground Ball Gets',
                   'Marks', 'Contested Marks', 
                   'Clearances', 'CB Clearance', 
                   'Player Pressure Acts', 'Player Pressure Points', 'Tackles', 'one_percenters',
                   'Intercept Marks', 'Spoils', 'Rebound 50\'s',
                   'Ruck Contests', 'Hit Outs', 'Hit Outs to Advantage', 
                   'clangers' ]


# Function to train and evaluate the Random Forest model with optimized hyperparameters
def train_evaluate_rf(X_train, y_train, X_test, y_test, best_params):
    rf = RandomForestRegressor(**best_params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, rmse, r2, rf

# Objective function for Optuna
def objective(trial):
    # Define the hyperparameters to be tuned
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 100),  # -1 means no limit
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        #'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'random_state': 0
    }

    rf = RandomForestRegressor(**param)
    
    # Perform cross-validation
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    mean_score = -np.mean(scores)  # Convert negative MSE back to positive
    
    return mean_score

# Load and prepare the data
# Assuming bf_data and columns_all are already defined

# One-hot encode the position feature
one_hot_encoder = OneHotEncoder()
encoded_positions = one_hot_encoder.fit_transform(bf_data[['POS']]).toarray()
encoded_position_df = pd.DataFrame(encoded_positions, columns=one_hot_encoder.get_feature_names_out(['POS']))

bf_data = bf_data.reset_index(drop=True)
encoded_position_df = encoded_position_df.reset_index(drop=True)

# Concatenate the one-hot encoded positions with the original data
bf_data = pd.concat([bf_data.drop(columns=['POS']), encoded_position_df], axis=1)

# Update the columns_all variable to include the one-hot encoded position columns
columns_all = globals()['columns_all'] + encoded_position_df.columns.tolist()

# Select features and target
X = bf_data[columns_all]
y = bf_data['Scaled Votes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Ensure all inputs are numpy arrays and handle missing data
X_train = X_train.fillna(0).values
y_train = y_train.fillna(0).values
X_test = X_test.fillna(0).values
y_test = y_test.fillna(0).values

# Create the study and optimize hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

# Get the best hyperparameters
best_params = study.best_params

# Train and evaluate the Random Forest model with the best hyperparameters
try:
    mae, mse, rmse, r2, rf = train_evaluate_rf(X_train, y_train, X_test, y_test, best_params)
    print(f"R²: {r2}, MAE: {mae}, RMSE: {rmse}")

    # Store feature importances
    importances = rf.feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    
    # Store evaluation metrics
    metrics = pd.DataFrame({
        'R2': [r2],
        'MAE': [mae],
        'RMSE': [rmse]
    })

    # Save the feature importances and metrics to CSV files
    feature_importances.to_csv('data/weights_df_5.csv', index=False)
    metrics.to_csv('data/metrics_df_5.csv', index=False)

    # Display the final feature importances and metrics dataframes
    print("Feature Importances:")
    print(feature_importances)
    print("\nEvaluation Metrics:")
    print(metrics)

except Exception as e:
    print(f"Error during training: {e}")










