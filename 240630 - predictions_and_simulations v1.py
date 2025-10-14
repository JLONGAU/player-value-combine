# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:42:16 2024

@author: Birch Matthew
"""


#%%

### Data prep for ML models
    
#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

### Load data

#Import CSVs
player_df = pd.read_csv("data/player_df.csv")
predicted_pav_df = pd.read_csv("results/predicted_pav_df.csv")

### Create features for ML

#If debut year is null, take the first season they played
player_min_season = player_df.groupby('Player_ID')['SEASON'].transform('min')
player_df['Debut_Year'] = player_df.apply(lambda row: player_min_season[row.name] if pd.isnull(row['Debut_Year']) else row['Debut_Year'], axis=1)

#Tag which players started before the start of the data
start_before_df = player_df[player_df['SEASON'] == player_df['SEASON'].min()][['Player_ID', 'games_prior']]
start_before_df = start_before_df[start_before_df['games_prior'] > 0]
start_before_df['Started Before First Year'] = True

player_df = pd.merge(player_df,
                     start_before_df[['Player_ID', 'Started Before First Year']],
                     how = 'left',
                     on = 'Player_ID'
                     )

player_df['Started Before First Year'] = player_df['Started Before First Year'].fillna(False)
    
#Tag which players who ended before the end of the data
ended_before_df = player_df[player_df['SEASON'] == player_df['SEASON'].max()][['Player_ID', 'games_season']]
ended_before_df = ended_before_df[ended_before_df['games_season'] > 0]
ended_before_df['Ended Before Last Year'] = False

player_df = pd.merge(player_df,
                     ended_before_df[['Player_ID', 'Ended Before Last Year']],
                     how = 'left',
                     on = 'Player_ID'
                     )

player_df['Ended Before Last Year'] = player_df['Ended Before Last Year'].fillna(True)
    
#Determine whether the full players career is in the data
player_df['Full Career In Data'] = ~(player_df['Started Before First Year'] | (player_df['Ended Before Last Year'] == False))

#Join on the PAV value
player_df = pd.merge(player_df,
                     predicted_pav_df[['SEASON', 'Player_ID', 'PAV_Total']],
                     how = 'left',
                     on = ['Player_ID', 'SEASON']
                    )

#Sort by player and season
player_df = player_df.sort_values(by=['Player_ID', 'SEASON'])

#Calculate cumulative PAV for each player
player_df['PAV_Cumulative'] = player_df.groupby('Player_ID')['PAV_Total'].cumsum()

#Calculate the number of years played
player_df['Years_Played'] = player_df.groupby('Player_ID').apply(lambda x: 1 + x['SEASON'] - x['SEASON'].min()).reset_index(level=0, drop=True)

#Save to CSV
player_df.to_csv('data/ml_data.csv')


#%%

### Visualisation
   
#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSVs
ml_data = pd.read_csv("data/ml_data.csv")

### Player Value by age

# Filter rows where full_career equals 1
filtered_data = ml_data[ml_data['Full Career In Data'] == 1]

# Create the 'years_played_cut' column by categorizing 'years_played' into bins
bins = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 100]
labels = ['1-2', '3-4', '5-6', '7-8', '9-10', '11-12', '13-14', '15-16', '17-18', '19+']
filtered_data['Years_Played_Category'] = pd.cut(filtered_data['Years_Played'], bins=bins, labels=labels, right=False)

# Plot using seaborn
sns.boxplot(x='Years_Played_Category', y='PAV_Total', data=filtered_data)
plt.xticks(rotation=45)  # Rotate labels to avoid overlap
plt.show()

### Cumulative Player value by age

# Filter the DataFrame for players who started after 2003 and were taken in the National draft
filtered_data = ml_data[ml_data['Started Before First Year'] == False]
filtered_data = filtered_data[filtered_data['Draft_Type'] == 'National']

# Define a colormap. Here, we'll use the 'viridis' colormap, which is a good perceptually uniform colormap in Matplotlib.
colormap = plt.cm.viridis

# Normalize pick values to match to the colormap
norm = mcolors.Normalize(vmin=filtered_data['Pick_Number'].min(), vmax=filtered_data['Pick_Number'].max())

# Plotting
fig, ax = plt.subplots()

for key, grp in filtered_data.groupby(['Player_ID']):
    # Map 'pick' values to colors
    color = colormap(norm(grp['Pick_Number'].iloc[0]))
    ax.plot(grp['age'], grp['PAV_Cumulative'], label=key, color=color)

# Create a colorbar for the 'pick' value
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Pick Number')

plt.xlabel('age')
plt.ylabel('Cumulative PAV')
plt.title('Cumulative PAV by Age for Natioanl Draft Players Who Started After 2003')
plt.show()

### Cumulative Games by player age

# Filter the DataFrame for full_career == 1
filtered_data = ml_data[ml_data['Full Career In Data'] == 1]

# Create the plot
fig, ax = plt.subplots()

# Normalize 'pick' values for color mapping
norm = plt.Normalize(filtered_data['Pick_Number'].min(), filtered_data['Pick_Number'].max())

# Choose a colormap
colormap = plt.cm.viridis

# Scatter plot
sc = ax.scatter(x=filtered_data['age'], 
                y=filtered_data['games_prior'] + filtered_data['games_season'], 
                c=filtered_data['Pick_Number'], 
                cmap=colormap, 
                norm=norm)

# Create colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Pick Number')

# Set labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Cumulative Games')
ax.set_title('Cumulative Games by Age for Players with Full Career')

plt.show()

#%% 

### Feature Engineering for ML models

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import numpy as np
import warnings
from itertools import product

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSV
ml_data = pd.read_csv("data/ml_data.csv")
squad_matches_data = pd.read_csv("data/squad_matches_df.csv")

#Select columns
ml_data = ml_data[['Player_ID', 'SEASON', 'age', 'POS', 'Years_Played', 'player_team',
                   'games_prior', 'games_season',
                   'PAV_Total', 'PAV_Cumulative',
                   'Started Before First Year', 'Draft_Type', 'Pick_Number', 'Is_Father_Son']]

#To add: Height, weight, Full Career In Data, Ended Before Last Year

##Other potential features to add: 
    #Top/mid/bottom 6 club, games missed due to injuries,
    #Cumulative count of positions
    #Cumulative count of teams
    #Previous (2,3,4) years PAV / Games
    #Height, BMI, home state, is home state where moved to
    #Cumulative brownlow, All Australians, fantasy points
    #Next 1, 3, 5 years PAV

# Cross join age range with all player IDs
player_ids = ml_data['Player_ID'].unique()
ages = range(16, 46)  # 16 to 45 inclusive
expanded_grid = pd.DataFrame(product(player_ids, ages), columns=['Player_ID', 'age'])

# Left join
ml_data_retirement_row = expanded_grid.merge(ml_data, on=['Player_ID', 'age'], how='left')

# Filter for players whose careers started after the start of the data
player_started_within_data = ml_data[ml_data['Started Before First Year'] == 0]['Player_ID'].unique()
ml_data_retirement_row = ml_data_retirement_row[ml_data_retirement_row['Player_ID'].isin(player_started_within_data)]

# Group by Player_ID and create max_age
grouped = ml_data_retirement_row.groupby('Player_ID')
max_age = grouped.apply(lambda x: x.loc[x['PAV_Total'].notna(), 'age'].max()).reset_index(name='max_age')
ml_data_retirement_row =  ml_data_retirement_row.merge(max_age, on='Player_ID')

# Group by Player_ID and create min_age
min_age = grouped.apply(lambda x: x.loc[x['PAV_Total'].notna(), 'age'].min()).reset_index(name='min_age')
ml_data_retirement_row =  ml_data_retirement_row.merge(min_age, on='Player_ID')

# Group by Player_ID and create min_season
min_season = grouped.apply(lambda x: x.loc[x['PAV_Total'].notna(), 'SEASON'].min()).reset_index(name='min_season')
ml_data_retirement_row =  ml_data_retirement_row.merge(min_season, on='Player_ID')

#Filter for all the rows that the player played, plus the next year after they registered their last game (i.e. retired)
ml_data_retirement_row = ml_data_retirement_row[
      (ml_data_retirement_row['age'] >= ml_data_retirement_row['min_age'])
    & (ml_data_retirement_row['age'] <= ml_data_retirement_row['max_age'] + 1)
    ]

# Fill missing 'season' values
ml_data_retirement_row['SEASON'] = ml_data_retirement_row['age'] - ml_data_retirement_row['min_age'] + ml_data_retirement_row['min_season']

#Sort by season
ml_data_retirement_row = ml_data_retirement_row.sort_values(by=['Player_ID', 'SEASON'])

#If PAV total and games season is null, then set to zero
ml_data_retirement_row['PAV_Total'].fillna(0, inplace=True)
ml_data_retirement_row['games_season'].fillna(0, inplace=True)

# Fill missing Years Played values
ml_data_retirement_row['Years_Played'] = ml_data_retirement_row['age'] - ml_data_retirement_row['min_age'] + 1

#Fill missing column values
def fill_columns(group):
    
    #Forward/backfill the draft type, pick number, position, and team name
    group['Draft_Type'] = group['Draft_Type'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    group['Pick_Number'] = group['Pick_Number'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    group['Is_Father_Son'] = group['Is_Father_Son'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    group['POS'] = group['POS'].transform(lambda x: x.ffill().bfill())
    group['player_team'] = group['player_team'].transform(lambda x: x.ffill().bfill())
    
    #Calculate cumulative PAV and games
    group['PAV_Cumulative'] = group['PAV_Total'].cumsum()
    group['Games_Cumulative'] = group['games_season'].cumsum()
    
    return group

ml_data_retirement_row = ml_data_retirement_row.groupby('Player_ID').apply(fill_columns)

#Reset the index
ml_data_retirement_row = ml_data_retirement_row.reset_index(drop=True)

# Create features for the number of matches a squad has played in each season
ml_data_retirement_row = pd.merge(ml_data_retirement_row,
                                  squad_matches_data,
                                  how = 'left',
                                  on = ['SEASON', 'player_team'])

#Features - Cumulative PAV and games 
ml_data_retirement_row['last_seasons_cumulative_PAV'] = ml_data_retirement_row.groupby('Player_ID')['PAV_Cumulative'].shift(1)
ml_data_retirement_row['last_seasons_cumulative_games'] = ml_data_retirement_row.groupby('Player_ID')['Games_Cumulative'].shift(1)

#Features - PAV in the last 1, 2, 3 and 4 seasons
ml_data_retirement_row['last_seasons_PAV'] = ml_data_retirement_row.groupby('Player_ID')['PAV_Total'].shift(1)
ml_data_retirement_row['last_two_seasons_PAV'] = ml_data_retirement_row.groupby('Player_ID')['PAV_Total'].transform(lambda x: x.rolling(window=2, min_periods=1).sum().shift())
ml_data_retirement_row['last_three_seasons_PAV'] = ml_data_retirement_row.groupby('Player_ID')['PAV_Total'].transform(lambda x: x.rolling(window=3, min_periods=1).sum().shift())
ml_data_retirement_row['last_four_seasons_PAV'] = ml_data_retirement_row.groupby('Player_ID')['PAV_Total'].transform(lambda x: x.rolling(window=4, min_periods=1).sum().shift())

#Features - Games played in the last 1, 2, 3 and 4 seasons
ml_data_retirement_row['last_seasons_games'] = ml_data_retirement_row.groupby('Player_ID')['games_season'].shift(1)
ml_data_retirement_row['last_two_seasons_games'] = ml_data_retirement_row.groupby('Player_ID')['games_season'].transform(lambda x: x.rolling(window=2, min_periods=1).sum().shift())
ml_data_retirement_row['last_three_seasons_games'] = ml_data_retirement_row.groupby('Player_ID')['games_season'].transform(lambda x: x.rolling(window=3, min_periods=1).sum().shift())
ml_data_retirement_row['last_four_seasons_games'] = ml_data_retirement_row.groupby('Player_ID')['games_season'].transform(lambda x: x.rolling(window=4, min_periods=1).sum().shift())

columns_to_fill = ['last_seasons_cumulative_PAV', 'last_seasons_cumulative_games',
                   'last_seasons_PAV', 'last_two_seasons_PAV', 'last_three_seasons_PAV', 'last_four_seasons_PAV',
                   'last_seasons_games', 'last_two_seasons_games', 'last_three_seasons_games', 'last_four_seasons_games',
                   'SQUAD_Matches_Played',
                   'last_seasons_squad_games', 'last_two_seasons_squad_games', 'last_three_seasons_squad_games', 'last_four_seasons_squad_games']

ml_data_retirement_row[columns_to_fill] = ml_data_retirement_row[columns_to_fill].fillna(0)

#Features - Percentage of games played
ml_data_retirement_row['seasons_games_pct']            = ml_data_retirement_row['games_season']             / ml_data_retirement_row['SQUAD_Matches_Played']
ml_data_retirement_row['last_seasons_games_pct']       = ml_data_retirement_row['last_seasons_games']       / ml_data_retirement_row['last_seasons_squad_games']
ml_data_retirement_row['last_two_seasons_games_pct']   = ml_data_retirement_row['last_two_seasons_games']   / ml_data_retirement_row['last_two_seasons_squad_games']
ml_data_retirement_row['last_three_seasons_games_pct'] = ml_data_retirement_row['last_three_seasons_games'] / ml_data_retirement_row['last_three_seasons_squad_games']
ml_data_retirement_row['last_four_seasons_games_pct']  = ml_data_retirement_row['last_four_seasons_games']  / ml_data_retirement_row['last_four_seasons_squad_games']

#Apply corrections for players with less than 4 years under their belt
# If Years_Played is 2, then [last_three_seasons_games_pct, last_four_seasons_games_pct] is equal to last_seasons_games_pct
ml_data_retirement_row.loc[ml_data_retirement_row['Years_Played'] == 2, ['last_two_seasons_games_pct', 'last_three_seasons_games_pct', 'last_four_seasons_games_pct']] = \
    ml_data_retirement_row.loc[ml_data_retirement_row['Years_Played'] == 2, 'last_seasons_games_pct']

# If Years_Played is 3, then [last_three_seasons_games_pct, last_four_seasons_games_pct] is equal to last_two_seasons_games_pct
ml_data_retirement_row.loc[ml_data_retirement_row['Years_Played'] == 3, ['last_three_seasons_games_pct', 'last_four_seasons_games_pct']] = \
    ml_data_retirement_row.loc[ml_data_retirement_row['Years_Played'] == 3, 'last_two_seasons_games_pct']

# If Years_Played is 4, then [last_four_seasons_games_pct] is equal to last_three_seasons_games_pct
ml_data_retirement_row.loc[ml_data_retirement_row['Years_Played'] == 4, 'last_four_seasons_games_pct'] = ml_data_retirement_row.loc[ml_data_retirement_row['Years_Played'] == 4, 'last_three_seasons_games_pct']

# Calculate PAV per game for the last 4 seasons
ml_data_retirement_row['PAV_per_game_season'] = ml_data_retirement_row.apply(
    lambda row: row['PAV_Total'] / row['games_season'] if row['games_season'] > 0 else 0, axis=1)

ml_data_retirement_row['last_seasons_PAV_per_game'] = ml_data_retirement_row.apply(
    lambda row: row['last_seasons_PAV'] / row['last_seasons_games'] if row['last_seasons_games'] > 0 else 0, axis=1)

ml_data_retirement_row['last_two_seasons_PAV_per_game'] = ml_data_retirement_row.apply(
    lambda row: row['last_two_seasons_PAV'] / row['last_two_seasons_games'] if row['last_two_seasons_games'] > 0 else 0, axis=1)

ml_data_retirement_row['last_three_seasons_PAV_per_game'] = ml_data_retirement_row.apply(
    lambda row: row['last_three_seasons_PAV'] / row['last_three_seasons_games'] if row['last_three_seasons_games'] > 0 else 0, axis=1)

ml_data_retirement_row['last_four_seasons_PAV_per_game'] = ml_data_retirement_row.apply(
    lambda row: row['last_four_seasons_PAV'] / row['last_four_seasons_games'] if row['last_four_seasons_games'] > 0 else 0, axis=1)

#Drop rows where values are null or infinity
columns_to_check = ['last_seasons_games_pct', 'last_two_seasons_games_pct', 'last_three_seasons_games_pct', 
                    'last_four_seasons_games_pct']

ml_data_retirement_row = ml_data_retirement_row.replace([np.inf, -np.inf], np.nan).dropna(subset=columns_to_check)

#Features - Games played in each position

#Map to simplified positions
position_mapping = {
    'Gen Def': 'DEF',
    'Gen Fwd': 'FWD',
    'Mid': 'MID',
    'Key Def': 'DEF',
    'Ruck': 'RUCK',
    'Mid-Fwd': 'MID',
    'Key Fwd': 'FWD',
    'Wing': 'WING'
}

ml_data_retirement_row['POS_Simplified'] = ml_data_retirement_row['POS'].map(position_mapping)

#Unique positions
positions = ml_data_retirement_row['POS_Simplified'].unique()

#Calculate cumulative games for each position
for pos in positions:
    ml_data_retirement_row[f'{pos}_games'] = ml_data_retirement_row.apply(lambda row: row['games_season'] if row['POS_Simplified'] == pos else 0, axis=1)
    ml_data_retirement_row[f'{pos}_cumulative_games'] = ml_data_retirement_row.groupby('Player_ID')[f'{pos}_games'].cumsum()
    
    ml_data_retirement_row[f'last_seasons_{pos}_games'] = ml_data_retirement_row.groupby('Player_ID')[f'{pos}_games'].shift(1).fillna(0)
    ml_data_retirement_row[f'last_seasons_{pos}_cumulative_games'] = ml_data_retirement_row.groupby('Player_ID')[f'{pos}_cumulative_games'].shift(1).fillna(0)


#Features - Draft type and pick number
ml_data_retirement_row['Is_Draft_National']   = ml_data_retirement_row['Draft_Type'].apply(lambda x: 1 if x == 'National' else 0)
ml_data_retirement_row['Is_Draft_Rookie']     = ml_data_retirement_row['Draft_Type'].apply(lambda x: 1 if x == 'Rookie' else 0)
ml_data_retirement_row['Is_Draft_Pre_Season'] = ml_data_retirement_row['Draft_Type'].apply(lambda x: 1 if x == 'Pre-Season' else 0)
ml_data_retirement_row['Is_Draft_Mid_Season'] = ml_data_retirement_row['Draft_Type'].apply(lambda x: 1 if x == 'Mid-Season' else 0)

ml_data_retirement_row['Pick_Number_National']   = ml_data_retirement_row.apply(lambda x: x['Pick_Number'] if x['Draft_Type'] == 'National'   else 999, axis=1)
ml_data_retirement_row['Pick_Number_Rookie']     = ml_data_retirement_row.apply(lambda x: x['Pick_Number'] if x['Draft_Type'] == 'Rookie'     else 999, axis=1)
ml_data_retirement_row['Pick_Number_Pre_Season'] = ml_data_retirement_row.apply(lambda x: x['Pick_Number'] if x['Draft_Type'] == 'Pre-Season' else 999, axis=1)
ml_data_retirement_row['Pick_Number_Mid_Season'] = ml_data_retirement_row.apply(lambda x: x['Pick_Number'] if x['Draft_Type'] == 'Mid-Season' else 999, axis=1)


#Features - Number of times a played has been traded

# Function to calculate cumulative trades
def calculate_cumulative_trades(player_group):
    unique_clubs = []
    trades = 0
    cumulative_trades = []
    previous_club = None
    for club in player_group:
        if club != previous_club:
            if club in unique_clubs:
                trades += 1  # Increment trade count if moving back to a previous club
            else:
                unique_clubs.append(club)
                trades += 1 if previous_club is not None else 0
        cumulative_trades.append(trades)
        previous_club = club
    return cumulative_trades

# Apply the function to each player's group and create the new column
ml_data_retirement_row['Player_cumulative_trades'] = ml_data_retirement_row.groupby('Player_ID')['player_team'].transform(calculate_cumulative_trades)

# Set when a player has retired 
ml_data_retirement_row['retired'] = np.where(ml_data_retirement_row['age'] > ml_data_retirement_row['max_age'] , "retire", "playing")

# Determine retirement status
retirement_status = ml_data_retirement_row.groupby('Player_ID')['retired'].apply(lambda x: any(item == "retire" for item in x)).astype(int)

# Map this retirement status back to the original dataframe
ml_data_retirement_row['Is_Retired'] = ml_data_retirement_row['Player_ID'].map(retirement_status)

#Drop games prior as we now use Games Cumulative 
ml_data_retirement_row.drop(columns = ['games_prior'], inplace = True)

#Remove 2024 as we dont have data for it yet
ml_data_retirement_row = ml_data_retirement_row[ml_data_retirement_row['SEASON'] != 2024]

#Save to csv
ml_data_retirement_row.to_csv('data/ml_data_features.csv', index = False)


#%%

### MODEL1: Retirement prediction

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings
import numpy as np

import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve

import matplotlib.pyplot as plt
import shap
import pickle

import mlflow
from mlflow.models import infer_signature
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSV
ml_data = pd.read_csv("data/ml_data_features.csv")

### Model fit

# Preparing the features and target
X = ml_data[[#Features - Demographics
             'age', 'Years_Played', 
             
             #Features - Draft & trade
             'Pick_Number_National', 'Pick_Number_Rookie', 'Pick_Number_Pre_Season', 'Pick_Number_Mid_Season',
             'Player_cumulative_trades', 'Is_Father_Son',
            
             #Features - Value provided
             'last_seasons_cumulative_PAV',
             'last_seasons_PAV', 'last_two_seasons_PAV', 'last_three_seasons_PAV', 'last_four_seasons_PAV', 
             #'last_seasons_PAV_per_game', 'last_two_seasons_PAV_per_game', 'last_three_seasons_PAV_per_game', 'last_four_seasons_PAV_per_game',
            
             #Features - Games played
             'last_seasons_cumulative_games',
             'last_seasons_games_pct', 'last_two_seasons_games_pct', 'last_three_seasons_games_pct', 'last_four_seasons_games_pct',
             'last_seasons_FWD_games', 'last_seasons_DEF_games', 'last_seasons_MID_games', 'last_seasons_WING_games', 'last_seasons_RUCK_games',
             'last_seasons_FWD_cumulative_games', 'last_seasons_DEF_cumulative_games', 'last_seasons_MID_cumulative_games', 'last_seasons_WING_cumulative_games', 'last_seasons_RUCK_cumulative_games'
             
             ]]

            #Features to add: Team performance, injury, contract status, player 

selected_columns = X.columns.tolist()
columns_df = pd.DataFrame(selected_columns, columns=['feature_names'])
columns_df.to_csv('data/retirement_model_feature_names.csv', index=False)

# Ensure 'retired' is numeric
y = ml_data['retired'].astype('category').cat.codes  

# Splitting the dataset into training and holdout (test) sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Define the optimization function
def objective(trial):
    # Define the hyperparameters to be tuned
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    
    # Define the model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('randomforestclassifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Perform cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=4)
    mean_score = np.mean(scores)
    
    return mean_score

# Create the study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best hyperparameters found
best_params = study.best_trial.params
best_score = study.best_trial.value
print(f"Best hyperparameters: {best_params}")
print(f"Best score: {best_score}")

# Extract the parameters for RandomForestClassifier
best_model_params = {
    'n_estimators': best_params['n_estimators'],
    'max_depth': best_params['max_depth'],
    'min_samples_split': best_params['min_samples_split'],
    'min_samples_leaf': best_params['min_samples_leaf'],
    'max_features': best_params['max_features'],
    'bootstrap': best_params['bootstrap'],
    'class_weight': 'balanced',
    'random_state': 42
}

# Instantiate and fit the best model
best_model = RandomForestClassifier(**best_model_params)
best_model.fit(X_train, y_train)

# Save the model to disk
with open('model/model_retire_predict.pkl', 'wb') as file:
    pickle.dump(best_model, file)

### Model evaluation

# Make predictions on the holdout set
prob_retire = best_model.predict_proba(X_test)[:, 1]

# Generate binary predictions based on the probability threshold of 0.5
y_pred = (prob_retire > 0.8).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, prob_retire)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy}')
print(f'ROC AUC: {roc_auc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, prob_retire)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal

# Add decision threshold labels to the curve
nth_thresholds = 3
for i, threshold in enumerate(thresholds[::nth_thresholds]):
    plt.text(fpr[i*nth_thresholds], tpr[i*nth_thresholds], s=round(threshold, 2), 
             fontdict={'size': 8},
             bbox=dict(facecolor='white', alpha=0.5))

# Plot 1: ROC Curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plot_path_1 = "figures/reirement_model_ROC.png"
plt.savefig(plot_path_1)
plt.show()
plt.close()

#PLot 2: Shap values

# For tree-based models like RandomForest, XGBoost, LightGBM, CatBoost
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(24, 8)) 
shap.summary_plot(shap_values, X_test, show=False) 
plot_path_2 = "figures/retirement_model_SHAP.png"
plt.savefig(plot_path_2, bbox_inches='tight')
plt.show()
plt.close()


#Plot 3: Precision recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, prob_retire)

# Calculate F1 scores at each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

# Find the threshold that maximizes F1 score
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]
best_f1_score = f1_scores[best_threshold_index]

# Find the intersection point (where precision and recall are closest)
intersection_index = np.argmin(np.abs(precisions - recalls))
intersection_threshold = thresholds[intersection_index]
intersection_precision = precisions[intersection_index]
intersection_recall = recalls[intersection_index]

print(f'Best Threshold: {best_threshold}')
print(f'Best F1 Score: {best_f1_score}')
print(f'Intersection Threshold: {intersection_threshold}')
print(f'Intersection Precision: {intersection_precision}')
print(f'Intersection Recall: {intersection_recall}')


# Plot precision-recall trade-off with the best threshold marked
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], label='Precision', color='b')
plt.plot(thresholds, recalls[:-1], label='Recall', color='g')
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
plt.scatter(intersection_threshold, intersection_precision, color='purple', zorder=5, label='Intersection Point')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Trade-off')
plt.legend()
plt.grid(True)
plot_path_3 = "figures/retirement_model_precision_recall_tradeoff.png"
plt.savefig(plot_path_3)
plt.show()
plt.close()


#Log the result in ML Flow
# Create a new MLflow Experiment

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Retirement Model")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(best_model_params)

    # Log individual metrics
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("ROC AUC", roc_auc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    # Get the schemas of the model inputs and outputs
    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.log_param("best_params", best_params)
    mlflow.log_param("best_score", best_score)

    # Log the model   
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="retirement_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="Retirement Model"
    )
    
    #Create tags
    mlflow.set_tag("model_type", "Random Forest Classifier")
    mlflow.set_tag("description", "Predict whether a player will retire")
    
    #Save the plots
    mlflow.log_artifact(plot_path_1)
    mlflow.log_artifact(plot_path_2)
    mlflow.log_artifact(plot_path_3)

### To log into MLFlow
#cd C:\bcg-repos\Player-Value\
#mlflow ui


#%%

### MODEL2: Games played model

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings
import numpy as np

import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import mlflow
from mlflow.models import infer_signature

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSV
ml_data = pd.read_csv("data/ml_data_features.csv")

#Remove 'retired' rows
ml_data = ml_data[ml_data['retired'] == 'playing']

# Preparing the features and target
X = ml_data[[
             #Features - Demographics
             'age', 'Years_Played', 
             
             #Features - Draft & trade
             'Pick_Number_National', 'Pick_Number_Rookie', 'Pick_Number_Pre_Season', 'Pick_Number_Mid_Season',
             'Player_cumulative_trades', 'Is_Father_Son',
             
             #Features - Value provided
             'last_seasons_cumulative_PAV',
             'last_seasons_PAV', 'last_two_seasons_PAV', 'last_three_seasons_PAV', 'last_four_seasons_PAV', 
             'last_seasons_PAV_per_game', 'last_two_seasons_PAV_per_game', 'last_three_seasons_PAV_per_game', 'last_four_seasons_PAV_per_game',
             
             #Features - Games played
             'last_seasons_cumulative_games', 
             'last_seasons_games_pct', 'last_two_seasons_games_pct', 'last_three_seasons_games_pct', 'last_four_seasons_games_pct',
             'last_seasons_FWD_games', 'last_seasons_DEF_games', 'last_seasons_MID_games', 'last_seasons_WING_games', 'last_seasons_RUCK_games',
             'last_seasons_FWD_cumulative_games', 'last_seasons_DEF_cumulative_games', 'last_seasons_MID_cumulative_games', 'last_seasons_WING_cumulative_games', 'last_seasons_RUCK_cumulative_games'
             ]] 

             #Features to add: from interstate, injury

# Create a DataFrame to store the column names
selected_columns = X.columns.tolist()
columns_df = pd.DataFrame(selected_columns, columns=['feature_names'])
columns_df.to_csv('data/games_model_feature_names.csv', index=False)

y = ml_data['seasons_games_pct']

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Define the optimization function for Optuna
def objective(trial):
    # Define the hyperparameters to be tuned
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'boosting_type': 'gbdt',  # traditional Gradient Boosting Decision Tree
        'objective': 'regression',
        'max_depth': trial.suggest_int('max_depth', 3, 100),  # -1 means no limit
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': 0
    }

    model = LGBMRegressor(**param)
    
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=4)
    mean_score = -np.mean(scores)  # Convert negative MSE back to positive
    
    return mean_score

# Create the study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best hyperparameters found
best_params = study.best_trial.params
best_score = study.best_value
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

# Instantiate and fit the best model
best_model = LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)

# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Save the model to disk
with open('model/model_games_predict.pkl', 'wb') as file:
    pickle.dump(best_model, file)

###Assess the model

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
# Root Mean Squared Error
rmse = np.sqrt(mse)

# R^2 value
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) value: {r2}")


#Scatter plot showing actual vs predicted

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)  # Scatter plot of actual vs predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line with slope=1, intercept=0
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Cumulative Games Model: Actual vs Predicted")
plot_path_1 = "figures/games_model_actual_vs_predicted.png"
plt.savefig(plot_path_1)
plt.show()
plt.close()
    
#Residuals plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plot_path_2 = "figures/games_model_residuals_plot.png"
plt.savefig(plot_path_2)
plt.show()
plt.close()

#Distribution of residuals

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plot_path_3 = "figures/games_model_residuals_distribution.png"
plt.savefig(plot_path_3)
plt.show()
plt.close()

#Feature importances

feature_importances = best_model.feature_importances_
features = X_train.columns

plt.figure(figsize=(24, 8))
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Plot')
plot_path_4 = "figures/games_model_feature_importance.png"
plt.savefig(plot_path_4)
plt.show()
plt.close()


#Log the result in ML Flow
# Create a new MLflow Experiment
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Games Played Model")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(best_params)

    # Log individual metrics
    mlflow.log_metric("Mean Absolute Error", mae)
    mlflow.log_metric("Mean Squared Error", mse)
    mlflow.log_metric("Root Mean Squared Error", rmse)
    mlflow.log_metric("R2 Value", r2)

    # Get the schemas of the model inputs and outputs
    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.log_param("best_score", best_score)

    # Log the model   
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="games_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="Games Model"
    )
    
    #Create tags
    mlflow.set_tag("model_type", "LightGBM")
    mlflow.set_tag("description", "Predict the games played for an upcoming season")
    
    #Save the plots
    mlflow.log_artifact(plot_path_1)
    mlflow.log_artifact(plot_path_2)
    mlflow.log_artifact(plot_path_3)
    mlflow.log_artifact(plot_path_4)


### To log into MLFlow
#cd C:\bcg-repos\Player-Value\
#mlflow ui


#%%

### MODEL3: Value prediction model

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings
import numpy as np

import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import mlflow
from mlflow.models import infer_signature

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSV
ml_data = pd.read_csv("data/ml_data_features.csv")

#Remove 'retired' rows
ml_data = ml_data[ml_data['retired'] == 'playing']

# Preparing the features and target
X = ml_data[[#Features - Demographics
             'age', 'Years_Played', 
             
             #Features - Draft & trade
             'Pick_Number_National', 'Pick_Number_Rookie', 'Pick_Number_Pre_Season', 'Pick_Number_Mid_Season',
             'Player_cumulative_trades', 'Is_Father_Son',
             
             #Features - Value provided
             'last_seasons_cumulative_PAV', 
             'last_seasons_PAV', 'last_two_seasons_PAV', 'last_three_seasons_PAV', 'last_four_seasons_PAV',
             'last_seasons_PAV_per_game', 'last_two_seasons_PAV_per_game', 'last_three_seasons_PAV_per_game', 'last_four_seasons_PAV_per_game',
             
             #Features - Games played
             'last_seasons_cumulative_games', 
             'seasons_games_pct', 'last_seasons_games_pct', 'last_two_seasons_games_pct', 'last_three_seasons_games_pct', 'last_four_seasons_games_pct',
             
             'FWD_games', 'DEF_games', 'MID_games', 'WING_games', 'RUCK_games',
             'FWD_cumulative_games', 'DEF_cumulative_games', 'MID_cumulative_games', 'WING_cumulative_games', 'RUCK_cumulative_games'
             ]]

selected_columns = X.columns.tolist()
columns_df = pd.DataFrame(selected_columns, columns=['feature_names'])
columns_df.to_csv('data/value_model_feature_names.csv', index=False)

y = ml_data['PAV_per_game_season']

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Define the optimization function for Optuna
def objective(trial):
    # Define the hyperparameters to be tuned
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'boosting_type': 'gbdt',  # traditional Gradient Boosting Decision Tree
        'objective': 'regression',
        'max_depth': trial.suggest_int('max_depth', 3, 100),  # -1 means no limit
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 0
    }

    model = LGBMRegressor(**param)
    
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=4)
    mean_score = -np.mean(scores)  # Convert negative MSE back to positive
    
    return mean_score

# Create the study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best hyperparameters found
best_params = study.best_trial.params
best_score = study.best_value
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

# Instantiate and fit the best model
best_model = LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)

# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Save the model to disk
with open('model/model_PAV_predict.pkl', 'wb') as file:
    pickle.dump(best_model, file)

###Assess the model

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
# Root Mean Squared Error
rmse = np.sqrt(mse)

# R^2 value
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) value: {r2}")

#Plot 1: Scatter plot showing actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)  # Scatter plot of actual vs predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line with slope=1, intercept=0
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Cumulative Games Model: Actual vs Predicted")
plot_path_1 = "figures/value_model_actual_vs_predicted.png"
plt.savefig(plot_path_1)
plt.show()
plt.close()

#Plot 2: Residuals plot

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plot_path_2 = "figures/value_model_residuals_plot.png"
plt.savefig(plot_path_2)
plt.show()
plt.close()


#Plot 3: Distribution of residuals

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plot_path_3 = "figures/value_model_residuals_distribution.png"
plt.savefig(plot_path_3)
plt.show()
plt.close()

#Plot 4: Feature importances

feature_importances = best_model.feature_importances_
features = X_train.columns

plt.figure(figsize=(24, 8))
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Plot')
plot_path_4 = "figures/value_model_feature_importance.png"
plt.savefig(plot_path_4)
plt.show()
plt.close()



#Log the result in ML Flow
# Create a new MLflow Experiment
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Value Provided Model")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(best_params)

    # Log individual metrics
    mlflow.log_metric("Mean Absolute Error", mae)
    mlflow.log_metric("Mean Squared Error", mse)
    mlflow.log_metric("Root Mean Squared Error", rmse)
    mlflow.log_metric("R2 Value", r2)

    # Get the schemas of the model inputs and outputs
    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.log_param("best_score", best_score)

    # Log the model   
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="value_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="Value Model"
    )
    
    #Create tags
    mlflow.set_tag("model_type", "LightGBM")
    mlflow.set_tag("description", "Predict the value provided per game")
    
    #Save the plots
    mlflow.log_artifact(plot_path_1)
    mlflow.log_artifact(plot_path_2)
    mlflow.log_artifact(plot_path_3)
    mlflow.log_artifact(plot_path_4)



### To log into MLFlow
#cd C:\bcg-repos\Player-Value\
#mlflow ui




#%%

### Run draft pick trajectory simulations

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings
import numpy as np
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

# Load the models from disk
with open('model/model_games_predict.pkl', 'rb') as file:
    model_games_predict = pickle.load(file)

with open('model/model_PAV_predict.pkl', 'rb') as file:
    model_PAV_predict = pickle.load(file)

with open('model/model_retire_predict.pkl', 'rb') as file:
    model_retire_predict = pickle.load(file)

#Import the feature names for each model
retirement_features = pd.read_csv('data/retirement_model_feature_names.csv')
games_features = pd.read_csv('data/games_model_feature_names.csv')
value_features = pd.read_csv('data/value_model_feature_names.csv')

#Define a function to safely get previousl index data
def safe_get(df, idx, col):
    try:
        return df.loc[idx, col]
    except KeyError:
        return 0

games_per_season = 23

def simulate_career(pick, position, id, years, noise=True, sd_season_games_pct = 0.25, sd_PAV_per_game = 0.25):
    #pick = 20
    #position = 'MID'
    #years = 20
     
    # Create DataFrame with pick and age range
    pick_data = pd.DataFrame({#Set demographic features                          
                              'age': range(18, 18 + years + 1), 
                              'Years_Played': range(1, years + 2),
                              
                              #Set draft & trade features
                              'Pick_Number_National': pick, 
                              'Pick_Number_Rookie': 999, 
                              'Pick_Number_Pre_Season': 999, 
                              'Pick_Number_Mid_Season': 999, 
                              'Player_cumulative_trades': 0, 
                              'Is_Father_Son': 0, 
                              
                              #Initialise position features
                              'Position': position,
                              'FWD_games': 0, 
                              'DEF_games': 0,  
                              'MID_games': 0,  
                              'WING_games': 0,  
                              'RUCK_games': 0,
                              
                              'FWD_cumulative_games': 0, 
                              'DEF_cumulative_games': 0, 
                              'MID_cumulative_games': 0, 
                              'WING_cumulative_games': 0, 
                              'RUCK_cumulative_games': 0
                                                      
                              })
       
    # Initialise first year data for value and games features
    pick_data.loc[0, [
        
        #Features - Value provided
        'last_seasons_cumulative_PAV',
        'last_seasons_PAV', 'last_two_seasons_PAV', 'last_three_seasons_PAV', 'last_four_seasons_PAV', 
        'last_seasons_PAV_per_game', 'last_two_seasons_PAV_per_game', 'last_three_seasons_PAV_per_game', 'last_four_seasons_PAV_per_game',
       
        #Features - Games played
        'last_seasons_cumulative_games', 
        'last_seasons_games', 'last_two_seasons_games', 'last_three_seasons_games', 'last_four_seasons_games',
        'last_seasons_games_pct', 'last_two_seasons_games_pct', 'last_three_seasons_games_pct', 'last_four_seasons_games_pct',
        'last_seasons_FWD_games', 'last_seasons_DEF_games', 'last_seasons_MID_games', 'last_seasons_WING_games', 'last_seasons_RUCK_games',
        'last_seasons_FWD_cumulative_games', 'last_seasons_DEF_cumulative_games', 'last_seasons_MID_cumulative_games', 'last_seasons_WING_cumulative_games', 'last_seasons_RUCK_cumulative_games'
        
        ]] = 0
    
  
    # Iterate through each year
    for i in range(years + 1):
        #i = 0

        # Generate noise
        noise_games = noise * np.random.normal(0, sd_season_games_pct)
        noise_value = noise * np.random.normal(0, sd_PAV_per_game)
        
        # Predict games
        pick_data.loc[i, 'seasons_games_pct'] = model_games_predict.predict(pick_data.loc[[i], 
           games_features['feature_names'].tolist()] ) + noise_games
        
        #Ensure games is not negative
        pick_data.loc[i, 'seasons_games_pct'] = np.clip(pick_data.loc[i, 'seasons_games_pct'], 0, 1)

        #Update the games played by position
        pick_data.loc[i, 'games_season'] = pick_data.loc[i, 'seasons_games_pct'] * games_per_season
        pick_data.loc[i, f'{position}_games'] = pick_data.loc[i, 'games_season']
        pick_data.loc[i, f'{position}_cumulative_games'] =  safe_get(pick_data, i-1, f'{position}_cumulative_games') + pick_data.loc[i, f'{position}_games']
        
        #Predict PAV
        pick_data.loc[i, 'PAV_per_game_season'] = model_PAV_predict.predict(pick_data.loc[[i], 
            value_features['feature_names'].tolist()]) + noise_value
        
        pick_data.loc[i, 'PAV_per_game_season'] = np.clip(pick_data.loc[i, 'PAV_per_game_season'], 0, 100)
        
        #Calculate total PAV, ensuring that the noise doesnt make the predictions negative
        pick_data.loc[i, 'PAV_total']    = pick_data.loc[i, 'PAV_per_game_season'] * pick_data.loc[i, 'games_season']
        
        # Update last seasons variables
        if i+1 in pick_data.index:
            
            #Features - Games total   
            pick_data.loc[i+1, 'last_seasons_games']       = safe_get(pick_data, i, 'games_season')
            pick_data.loc[i+1, 'last_two_seasons_games']   = safe_get(pick_data, i, 'games_season') + pick_data.loc[i, 'last_seasons_games'] 
            pick_data.loc[i+1, 'last_three_seasons_games'] = safe_get(pick_data, i, 'games_season') + pick_data.loc[i, 'last_two_seasons_games'] 
            pick_data.loc[i+1, 'last_four_seasons_games']  = safe_get(pick_data, i, 'games_season') + pick_data.loc[i, 'last_three_seasons_games'] 
                   
            #Features - Games cumulative
            pick_data.loc[i+1, 'last_seasons_cumulative_games'] = pick_data.loc[i, 'last_seasons_cumulative_games'] + pick_data.loc[i, 'games_season']
            
            #Features - Games pct
            pick_data.loc[i+1, 'last_seasons_games_pct']        = pick_data.loc[i+1, 'last_seasons_games']       / (1 * games_per_season)  
            pick_data.loc[i+1, 'last_two_seasons_games_pct']    = pick_data.loc[i+1, 'last_two_seasons_games']   / (2 * games_per_season)  
            pick_data.loc[i+1, 'last_three_seasons_games_pct']  = pick_data.loc[i+1, 'last_three_seasons_games'] / (3 * games_per_season)  
            pick_data.loc[i+1, 'last_four_seasons_games_pct']   = pick_data.loc[i+1, 'last_four_seasons_games']  / (4 * games_per_season)  

            #Features - Games by position
            pick_data.loc[i+1, 'last_seasons_FWD_games']  = pick_data.loc[i, 'FWD_games']
            pick_data.loc[i+1, 'last_seasons_DEF_games']  = pick_data.loc[i, 'DEF_games']
            pick_data.loc[i+1, 'last_seasons_MID_games']  = pick_data.loc[i, 'MID_games']
            pick_data.loc[i+1, 'last_seasons_WING_games'] = pick_data.loc[i, 'WING_games']
            pick_data.loc[i+1, 'last_seasons_RUCK_games'] = pick_data.loc[i, 'RUCK_games']
            
            #Features - Games by position cumulative
            pick_data.loc[i+1, 'last_seasons_FWD_cumulative_games']  = pick_data.loc[i, 'last_seasons_FWD_cumulative_games']  + pick_data.loc[i, 'FWD_games']
            pick_data.loc[i+1, 'last_seasons_DEF_cumulative_games']  = pick_data.loc[i, 'last_seasons_DEF_cumulative_games']  + pick_data.loc[i, 'DEF_games']
            pick_data.loc[i+1, 'last_seasons_MID_cumulative_games']  = pick_data.loc[i, 'last_seasons_MID_cumulative_games']  + pick_data.loc[i, 'MID_games']
            pick_data.loc[i+1, 'last_seasons_WING_cumulative_games'] = pick_data.loc[i, 'last_seasons_WING_cumulative_games'] + pick_data.loc[i, 'WING_games']
            pick_data.loc[i+1, 'last_seasons_RUCK_cumulative_games'] = pick_data.loc[i, 'last_seasons_RUCK_cumulative_games'] + pick_data.loc[i, 'RUCK_games']
            
            #Features - PAV cumulative
            pick_data.loc[i+1, 'last_seasons_cumulative_PAV'] = pick_data.loc[i, 'last_seasons_cumulative_PAV'] + pick_data.loc[i, 'PAV_total']
            
            #Features - PAV 
            pick_data.loc[i+1, 'last_seasons_PAV']            = safe_get(pick_data, i  , 'PAV_total')
            pick_data.loc[i+1, 'last_two_seasons_PAV']        = safe_get(pick_data, i  , 'PAV_total') + pick_data.loc[i, 'last_seasons_PAV']
            pick_data.loc[i+1, 'last_three_seasons_PAV']      = safe_get(pick_data, i  , 'PAV_total') + pick_data.loc[i, 'last_two_seasons_PAV']
            pick_data.loc[i+1, 'last_four_seasons_PAV']       = safe_get(pick_data, i  , 'PAV_total') + pick_data.loc[i, 'last_three_seasons_PAV']
            
            #Features - PAV per game       
            pick_data.loc[i+1, 'last_seasons_PAV_per_game']       = np.where(pick_data.loc[i+1, 'last_seasons_games']       == 0, 0,  pick_data.loc[i+1, 'last_seasons_PAV']       / pick_data.loc[i+1, 'last_seasons_games'])
            pick_data.loc[i+1, 'last_two_seasons_PAV_per_game']   = np.where(pick_data.loc[i+1, 'last_two_seasons_games']   == 0, 0,  pick_data.loc[i+1, 'last_two_seasons_PAV']   / pick_data.loc[i+1, 'last_two_seasons_games'])
            pick_data.loc[i+1, 'last_three_seasons_PAV_per_game'] = np.where(pick_data.loc[i+1, 'last_three_seasons_games'] == 0, 0,  pick_data.loc[i+1, 'last_three_seasons_PAV'] / pick_data.loc[i+1, 'last_three_seasons_games'])
            pick_data.loc[i+1, 'last_four_seasons_PAV_per_game']  = np.where(pick_data.loc[i+1, 'last_four_seasons_games']  == 0, 0,  pick_data.loc[i+1, 'last_four_seasons_PAV']  / pick_data.loc[i+1, 'last_four_seasons_games'])
            
        #Apply corrections for players with less than 4 years under their belt
        pick_data.loc[pick_data['Years_Played'] == 2, ['last_two_seasons_games_pct', 'last_three_seasons_games_pct', 'last_four_seasons_games_pct']] = \
            pick_data.loc[pick_data['Years_Played'] == 2, 'last_seasons_games_pct']

        # If Years_Played is 3, then [last_three_seasons_games_pct, last_four_seasons_games_pct] is equal to last_two_seasons_games_pct
        pick_data.loc[pick_data['Years_Played'] == 3, ['last_three_seasons_games_pct', 'last_four_seasons_games_pct']] = \
            pick_data.loc[pick_data['Years_Played'] == 3, 'last_two_seasons_games_pct']

        # If Years_Played is 4, then [last_four_seasons_games_pct] is equal to last_three_seasons_games_pct
        pick_data.loc[pick_data['Years_Played'] == 4, 'last_four_seasons_games_pct'] = pick_data.loc[pick_data['Years_Played'] == 4, 'last_three_seasons_games_pct']    

    # Simulate retirement event
    pick_data['retire_probability'] = [model_retire_predict.predict_proba(pick_data.loc[[i],
                               retirement_features['feature_names'].tolist()])[0][1] for i in range(years + 1)]
    
    # Ensure the player doesn't retire in the first simulated year
    pick_data['retire_probability'][0] = 0  
    
    # Function to determine if a player retires based on their probability
    def determine_retirement(row):
        return np.random.rand() < row['retire_probability']
    
    # Use a stochastic method to predict retirement
    pick_data['retire_binary'] = pick_data.apply(determine_retirement, axis=1).astype(int)
    
    #Use a threshold method to predict retirement
    #pick_data['retire_binary'] = (pick_data['retire_probability'] > 0.7).astype(int)
    
    pick_data['retire_final'] = pick_data['retire_binary'].cummax()
    pick_data['sim_id'] = id
    
    print(f'Position {position}, pick {pick}, run number {id}')
    
    return pick_data

###Run simulation

#Simulation parameters
simulate_picks = [i for i in range( 1,97,1)]
simulate_positions = ['DEF', 'FWD', 'MID', 'WING', 'RUCK']
simulate_runs = range(1, 51)
simulate_years = 20

"""
#For quicker test
simulate_picks = [i for i in range( 1,90,1)]
simulate_positions = ['MID']
simulate_runs = range(1, 10)
simulate_years = 20
"""

simulation_results = pd.concat([simulate_career(pick, position, x, simulate_years) for pick in simulate_picks for position in simulate_positions for x in simulate_runs], ignore_index=True)

# Plot the career path for pick 1
pick_1_data = simulation_results[simulation_results['Pick_Number_National'] == 1]
plt.figure(figsize=(10, 6))

# Use plt.scatter to directly create a mappable object
scatter = plt.scatter(data=pick_1_data, x='age', y='last_seasons_cumulative_PAV', c='retire_probability', cmap='viridis')

# Add the colorbar directly, using the mappable object
plt.colorbar(scatter, label='Probability of Retiring')

plt.xlim(18, 36)
plt.ylim(0, 250)
plt.ylabel('Value')
plt.xlabel('Age')
plt.title("Average Career trajectory for Pick 1 - color = probability of retiring")
plt.show()

#Save to csv
simulation_results.to_csv('results/career_simulation_results_pick.csv')


#%%

### Apply regression and plot simulation results

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import seaborn as sns

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

# Load the models from disk
simulation_results = pd.read_csv('results/career_simulation_results_pick.csv')

#Remove columns no longer needed
simulation_results = simulation_results[['Position', 'sim_id', 'Pick_Number_National', 'age', 'Years_Played', 'PAV_total', 'retire_final']]
        
# Sorting the data by Pick Number and Years Played to ensure the order is correct for rolling calculations
simulation_results = simulation_results.sort_values(by=["Position", 'Pick_Number_National', "sim_id", "Years_Played"])

#For each pick, calculate value over next 1 to 20 years
# To calculate future sums, we will shift the data backwards by the corresponding period minus one
years_to_sum = [i for i in range(1,21)]
for years in years_to_sum:
    simulation_results[f'PAV_{years}_years'] = (
        simulation_results.groupby(["Position", 'Pick_Number_National', "sim_id"], group_keys=False)  
        .apply(lambda x: x['PAV_total'].rolling(window=years, min_periods=1).sum().shift(-years + 1))
    )
       
#Count the number of years remaining for each player for each simulation
expected_years_until_retire = simulation_results[simulation_results['retire_final'] == 0].groupby(["Position", 'Pick_Number_National', "sim_id"]).size().reset_index(name='Exp_Years_Retire')
   
#Filter out for season 2024 only
simulation_results = simulation_results[simulation_results['Years_Played'] == 1]

#Join expected years until retire
simulation_results = pd.merge(simulation_results,
                              expected_years_until_retire,
                              how = 'left',
                              on = ["Position", 'Pick_Number_National', "sim_id"])

#Fill NAs with zeros
simulation_results['Exp_Years_Retire'] = simulation_results['Exp_Years_Retire'].fillna(0)

#Function to detemine expected PAV until retirement
def calculate_expected_pav(row):
    years_until_retire = row['Exp_Years_Retire']
    if years_until_retire == 0:
        return 0
    elif years_until_retire > 20:
        return row['PAV_20_years']
    else:
        # Retrieve the column name dynamically based on the value of Expected_Years_Until_Retire
        column_name = f"PAV_{int(years_until_retire)}_years"
        return row[column_name]

# Apply the function to each row in the DataFrame
simulation_results['Exp_PAV_Retire'] = simulation_results.apply(calculate_expected_pav, axis=1)

# Define the columns for which we need to calculate the median
median_columns = [f'PAV_{i}_years' for i in range(1,21)] + ['Exp_Years_Retire', 'Exp_PAV_Retire']

# Calculate the median for the specified columns
simulation_results_summary = simulation_results.groupby(['Pick_Number_National', "Position"])[median_columns].median().reset_index()

# Calculate quantiles by pick number
def quantiles(group):
    return pd.Series({
        'p10': group['Exp_PAV_Retire'].quantile(0.1),
        'p30': group['Exp_PAV_Retire'].quantile(0.3),
        'p50': group['Exp_PAV_Retire'].quantile(0.5),
        'p70': group['Exp_PAV_Retire'].quantile(0.7),
        'p90': group['Exp_PAV_Retire'].quantile(0.9)
    })

simulation_results_summary = pd.merge(simulation_results_summary,
                                      simulation_results.groupby(['Pick_Number_National', 'Position']).apply(quantiles).reset_index(),
                                      on = ['Pick_Number_National', 'Position'],
                                      how = 'left')

#Save to csv
simulation_results_summary.to_csv('results/career_simulation_results_pick_quantiles.csv', index = False)

#Create an empty dataframe for regression results
simulation_results_df_final = pd.DataFrame()

### Run an exponential regression to smooth out draft pick values

positions = list(simulation_results['Position'].unique())

for position in positions: 
    
    simulation_results_df_pos = simulation_results_summary[simulation_results_summary['Position'] == position]
    
    # Extract 'Pick_Number' and 'p50' column as X and Y
    X = simulation_results_df_pos['Pick_Number_National'].values.reshape(-1,1)
    Y = simulation_results_df_pos['p50'].values
    
    # Function to fit polynomial regression with custom weights and calculate Mean Squared Error
    def weighted_polynomial_regression(degree, X, Y):
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
    
        # Assign higher weights to lower pick numbers
        weights = 1 / X.flatten()  # Inversely proportional to the pick numbers
    
        # Split the dataset into training and testing sets along with weights
        X_train, X_test, Y_train, Y_test, weights_train, weights_test = train_test_split(
            X_poly, Y, weights, test_size=0.1, random_state=42)
    
        # Fit the regression model with weights
        poly_model = LinearRegression()
        poly_model.fit(X_train, Y_train, sample_weight=weights_train)
    
        # Predict and calculate metrics
        Y_pred = poly_model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred, sample_weight=weights_test)
        r2 = r2_score(Y_test, Y_pred, sample_weight=weights_test)
    
        return mse, r2, poly_model, poly_features
    
    # Test various polynomial degrees to find the best fit
    degrees = [6]   #[1, 2, 3, 4, 5, 6]
    results = []
    for degree in degrees:
        mse, r2, model, transformer = weighted_polynomial_regression(degree, X, Y)
        results.append((degree, mse, r2, model, transformer))
        
    # Find the best degree with the highest R^2
    best_degree, best_mse, best_r2, best_model, best_transformer = max(results, key=lambda item: item[2])
    
    # Print the best degree with its Mean Squared Error and R^2
    print(f"Best degree: {best_degree} with MSE: {best_mse} and R^2: {best_r2}")
    
    # We apply the transformation to the whole dataset and predict
    X_poly_all = best_transformer.transform(X)
    Y_pred_all = best_model.predict(X_poly_all)
    
    
    for i in range(1, len(Y_pred_all)):
        # Adjust predictions to ensure they are always at least slightly decreasing
        if Y_pred_all[i] >= Y_pred_all[i - 1] - 0.5:
            Y_pred_all[i] = Y_pred_all[i - 1] - 0.5
        #Ensure that pick value is never below zero
        Y_pred_all[i] = max(Y_pred_all[i], 0)
    
    # Add the prediction data as a new column in the original dataframe
    simulation_results_df_pos['predicted_p50'] = Y_pred_all
    
    # Plot 1: The original data and the best fit line with R^2 annotation
    plt.scatter(X, Y, color='blue', label='Data')
    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    Y_range_pred = best_model.predict(best_transformer.transform(X_range))
    plt.plot(X_range, Y_range_pred, color='red', label=f'Best Fit: Degree {best_degree} (R^2: {best_r2:.2f})')
    plt.title(f'Value by Pick - {position}')
    plt.ylim(0, 250)
    plt.xlabel('Pick Number')
    plt.ylabel('Value (p50)')
    plt.legend()
    plt.show()
    
    
    # Plot 2: Chart with confidence bands and the regression results
    
    # Set the seaborn style
    sns.set(style="whitegrid")
    
    # Creating the plot
    plt.figure(figsize=(10, 6))
    
    # Adjust the x-axis to show every second integer starting from 1
    plt.xticks(np.arange(1, max(simulation_results_df_pos['Pick_Number_National']) + 1, 5))
    
    # Geom_ribbon equivalent with fill_between for the p10 to p90 range
    plt.fill_between(x=simulation_results_df_pos['Pick_Number_National'], y1=simulation_results_df_pos['p10'], y2=simulation_results_df_pos['p90'], color='grey', alpha=0.4)
    
    # Geom_ribbon equivalent with fill_between for the p30 to p70 range
    plt.fill_between(x=simulation_results_df_pos['Pick_Number_National'], y1=simulation_results_df_pos['p30'], y2=simulation_results_df_pos['p70'], color='grey', alpha=0.8)
    
    # Geom_line equivalent for the median line (p50)
    plt.plot(simulation_results_df_pos['Pick_Number_National'], simulation_results_df_pos['p50'], color='black', linestyle='dashed', linewidth=2)
    
    # Regression line
    plt.plot(simulation_results_df_pos['Pick_Number_National'], simulation_results_df_pos['predicted_p50'], color='red', linestyle='dashed', linewidth=2)
    
    plt.ylabel('Value', fontsize = 12)
    plt.xlabel('Pick', fontsize = 12)
    plt.ylim(0, 250)
    plt.title(f'Value by Pick - {position}', fontsize = 16)
    
    # Add vertical spans for different rounds
    round_length = 18  # Assuming each round has 18 picks
    total_picks = max(simulation_results_df_pos['Pick_Number_National'])
    # Calculate the number of rounds
    num_rounds = (total_picks + round_length - 1) // round_length  
    # Generate a palette for desired color
    base_color = sns.color_palette("Blues", n_colors=num_rounds)  
    
    for i in range(num_rounds):
        start_pick = i * round_length
        end_pick = total_picks
        color = base_color[i]
        plt.axvspan(start_pick, end_pick, color=color, alpha=0.25, label=f'Round {i + 1}')
    
    # To ensure the legend does not repeat labels, only keep unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    #Save the image
    plot_path = f"figures/Value_by_pick_{position}.png"
    plt.savefig(plot_path)
    
    # Show plot
    plt.show()
    
    simulation_results_df_final =  pd.concat([simulation_results_df_pos, simulation_results_df_final], ignore_index=True)

#Scale all of the other results to match the regression smoothing
simulation_results_df_final['Scale_Factor'] = simulation_results_df_final['predicted_p50'] / simulation_results_df_final['p50']
columns_to_scale = [col for col in simulation_results_df_final.columns if 'PAV' in col]

for col in columns_to_scale:
    simulation_results_df_final[col] = simulation_results_df_final[col] * simulation_results_df_final['Scale_Factor']

# Save the results
simulation_results_df_final.to_csv('results/career_simulation_results_pick_quantiles.csv')



#%%


### Simulate for each player the value remaining

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

import pandas as pd
import warnings
import pickle
import numpy as np

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
warnings.filterwarnings("ignore")

#Import CSV
ml_data = pd.read_csv("data/ml_data_features.csv")

#Filter for players that are not retired
ml_data = ml_data[ml_data['Is_Retired'] == 0]

# Find the index of the most recent row for each Player_ID using the maximum "SEASON"
most_recent_indices = ml_data.groupby('Player_ID')['SEASON'].idxmax()

# Filter the dataframe using these indices
ml_data = ml_data.loc[most_recent_indices]

# Load the models from disk
with open('model/model_games_predict.pkl', 'rb') as file:
    model_games_predict = pickle.load(file)

with open('model/model_PAV_predict.pkl', 'rb') as file:
    model_PAV_predict = pickle.load(file)

with open('model/model_retire_predict.pkl', 'rb') as file:
    model_retire_predict = pickle.load(file)

#Import the feature names for each model
retirement_features = pd.read_csv('retirement_model_feature_names.csv')
games_features = pd.read_csv('games_model_feature_names.csv')
value_features = pd.read_csv('value_model_feature_names.csv')

#Define a function to safely get previousl index data
def safe_get(df, idx, col):
    try:
        return df.loc[idx, col]
    except KeyError:
        return 0

#Set the number of games per season
games_per_season = 23

#Function to simulate the career for a player
def simulate_career(#Features - Demographcis
                    Player_ID, player_team, SEASON, age, Years_Played, 
                    
                    #Features - Draft & trade
                    Pick_Number_National, Pick_Number_Rookie, Pick_Number_Pre_Season, Pick_Number_Mid_Season,
                    Player_cumulative_trades, Is_Father_Son,
                    
                    #Features - Value
                    last_seasons_cumulative_PAV, 
                    last_seasons_PAV, last_two_seasons_PAV, last_three_seasons_PAV, last_four_seasons_PAV, 
                    last_seasons_PAV_per_game, last_two_seasons_PAV_per_game, last_three_seasons_PAV_per_game, last_four_seasons_PAV_per_game,
                    
                    #Features - Games
                    last_seasons_cumulative_games, 
                    last_seasons_games, last_two_seasons_games, last_three_seasons_games, last_four_seasons_games,
                    last_seasons_games_pct, last_two_seasons_games_pct, last_three_seasons_games_pct, last_four_seasons_games_pct, 
                    FWD_games, DEF_games, MID_games, WING_games, RUCK_games,
                    FWD_cumulative_games, DEF_cumulative_games, MID_cumulative_games, WING_cumulative_games, RUCK_cumulative_games,
                    last_seasons_FWD_games, last_seasons_DEF_games, last_seasons_MID_games, last_seasons_WING_games, last_seasons_RUCK_games,
                    last_seasons_FWD_cumulative_games, last_seasons_DEF_cumulative_games, last_seasons_MID_cumulative_games, last_seasons_WING_cumulative_games, last_seasons_RUCK_cumulative_games,
                    
                    #Target
                    seasons_games_pct, games_season, PAV_per_game_season, PAV_total,
                    
                    #Other parameters
                    id, max_age=38, noise=True, sd_season_games_pct = 0.25, sd_PAV_per_game = 0.25):
    
    """
    #Set sample data
    Player_ID, player_team, SEASON, age, Years_Played = 'Max_gawn_1991', 'Melbourne', 2023.00, 31, 13
    Pick_Number_National, Pick_Number_Rookie, Pick_Number_Pre_Season, Pick_Number_Mid_Season, Player_cumulative_trades, Is_Father_Son = 34, 999, 999, 999, 0, 0
    last_seasons_cumulative_PAV, last_seasons_PAV, last_two_seasons_PAV, last_three_seasons_PAV, last_four_seasons_PAV = 259, 26.68200874, 61.31638985, 95.12223004, 132.8489116
    last_seasons_cumulative_games, last_seasons_games_pct, last_two_seasons_games_pct, last_three_seasons_games_pct, last_four_seasons_games_pct = 181, 0.916666667,	 0.959183673, 0.924242424	, 0.931818182
    FWD_games, DEF_games, MID_games, WING_games, RUCK_games = 0, 0, 0, 0, 22 
    FWD_cumulative_games, DEF_cumulative_games, MID_cumulative_games, WING_cumulative_games, RUCK_cumulative_games = 9, 0, 0, 0, 194
    seasons_games_pct, games_season, PAV_per_game_season, PAV_total = 0.88, 22, 1.050913888, 23.12010554
    last_seasons_PAV_per_game, last_two_seasons_PAV_per_game, last_three_seasons_PAV_per_game, last_four_seasons_PAV_per_game = 1.212818579,	 1.304604039, 1.55938082, 1.620108678
    last_seasons_FWD_games, last_seasons_DEF_games, last_seasons_MID_games, last_seasons_WING_games, last_seasons_RUCK_games = 0, 0, 0, 0, 22
    last_seasons_FWD_cumulative_games, last_seasons_DEF_cumulative_games, last_seasons_MID_cumulative_games, last_seasons_WING_cumulative_games, last_seasons_RUCK_cumulative_games = 9, 0, 0, 0, 172    
    last_seasons_games, last_two_seasons_games, last_three_seasons_games, last_four_seasons_games = 22, 47, 61, 82
    """
       
    # Create DataFrame with demographic and draft data
    player_data = pd.DataFrame({#Set demographic features     
                              'Player_ID': Player_ID, 'Player_Team': player_team,
                              'age': range(age, max_age + 1), 
                              'Years_Played': range(Years_Played, Years_Played + 1 + max_age - age),
                              
                              #Set draft & trade features
                              'Pick_Number_National': Pick_Number_National, 
                              'Pick_Number_Rookie': Pick_Number_Rookie, 
                              'Pick_Number_Pre_Season': Pick_Number_Pre_Season, 
                              'Pick_Number_Mid_Season': Pick_Number_Mid_Season, 
                              'Player_cumulative_trades': Player_cumulative_trades, 
                              'Is_Father_Son': Is_Father_Son,
                              
                              #Initialise position features
                              'FWD_games': 0, 
                              'DEF_games': 0,  
                              'MID_games': 0,  
                              'WING_games': 0,  
                              'RUCK_games': 0,
                              
                              'FWD_cumulative_games': FWD_cumulative_games, 
                              'DEF_cumulative_games': DEF_cumulative_games, 
                              'MID_cumulative_games': MID_cumulative_games, 
                              'WING_cumulative_games': WING_cumulative_games, 
                              'RUCK_cumulative_games': RUCK_cumulative_games
                                            
                              })
    
    # Initialise games played and value data for the season with data available
    player_data.loc[0, ['SEASON']] = SEASON
    
    #Features - Value
    player_data.loc[0, ['last_seasons_cumulative_PAV']] = last_seasons_cumulative_PAV
    player_data.loc[0, ['last_seasons_PAV']] = last_seasons_PAV
    player_data.loc[0, ['last_two_seasons_PAV']] = last_two_seasons_PAV
    player_data.loc[0, ['last_three_seasons_PAV']] = last_three_seasons_PAV
    player_data.loc[0, ['last_four_seasons_PAV']] = last_four_seasons_PAV
    
    player_data.loc[0, ['last_seasons_PAV_per_game']] = last_seasons_PAV_per_game
    player_data.loc[0, ['last_two_seasons_PAV_per_game']] = last_two_seasons_PAV_per_game
    player_data.loc[0, ['last_three_seasons_PAV_per_game']] = last_three_seasons_PAV_per_game
    player_data.loc[0, ['last_four_seasons_PAV_per_game']] = last_four_seasons_PAV_per_game   
    
    #Features - Games played
    player_data.loc[0, ['last_seasons_cumulative_games']] = last_seasons_cumulative_games
    player_data.loc[0, ['last_seasons_games']] = last_seasons_games
    player_data.loc[0, ['last_two_seasons_games']] = last_two_seasons_games
    player_data.loc[0, ['last_three_seasons_games']] = last_three_seasons_games
    player_data.loc[0, ['last_four_seasons_games']] = last_four_seasons_games

    player_data.loc[0, ['last_seasons_games_pct']] = last_seasons_games_pct
    player_data.loc[0, ['last_two_seasons_games_pct']] = last_two_seasons_games_pct
    player_data.loc[0, ['last_three_seasons_games_pct']] = last_three_seasons_games_pct
    player_data.loc[0, ['last_four_seasons_games_pct']] = last_four_seasons_games_pct
    
    player_data.loc[0, ['FWD_games']] = FWD_games
    player_data.loc[0, ['DEF_games']] = DEF_games
    player_data.loc[0, ['MID_games']] = MID_games
    player_data.loc[0, ['WING_games']] = WING_games
    player_data.loc[0, ['RUCK_games']] = RUCK_games
    
    player_data.loc[0, ['last_seasons_FWD_games']] = last_seasons_FWD_games
    player_data.loc[0, ['last_seasons_DEF_games']] = last_seasons_DEF_games
    player_data.loc[0, ['last_seasons_MID_games']] = last_seasons_MID_games
    player_data.loc[0, ['last_seasons_WING_games']] = last_seasons_WING_games
    player_data.loc[0, ['last_seasons_RUCK_games']] = last_seasons_RUCK_games
    player_data.loc[0, ['last_seasons_FWD_cumulative_games']] = last_seasons_FWD_cumulative_games
    player_data.loc[0, ['last_seasons_DEF_cumulative_games']] = last_seasons_DEF_cumulative_games
    player_data.loc[0, ['last_seasons_MID_cumulative_games']] = last_seasons_MID_cumulative_games
    player_data.loc[0, ['last_seasons_WING_cumulative_games']] = last_seasons_WING_cumulative_games
    player_data.loc[0, ['last_seasons_RUCK_cumulative_games']] = last_seasons_RUCK_cumulative_games

    #Determine the latest position of the player from the latest number of games in that position
    positions = ['FWD_games', 'DEF_games', 'MID_games', 'WING_games', 'RUCK_games'] 
    position = player_data[positions].iloc[0].idxmax().replace('_games', '')
    player_data['Position'] = position

    #Initialise target value
    player_data.loc[0, ['seasons_games_pct']] = seasons_games_pct
    player_data.loc[0, ['games_season']] = games_season
    player_data.loc[0, ['PAV_per_game_season']] = PAV_per_game_season
    player_data.loc[0, ['PAV_total']] = PAV_total
    

    # Iterate through each year
    for i in range(max_age + 1 - age):
        #i = 0
        
        # Generate noise
        noise_games = noise * np.random.normal(0, sd_season_games_pct)
        noise_value = noise * np.random.normal(0, sd_PAV_per_game)
        
        #If a prediction is needed
        if pd.isna(player_data.loc[i, 'seasons_games_pct']):  
            #i = 1
            # Predict games
            player_data.loc[i, 'seasons_games_pct'] = model_games_predict.predict(player_data.loc[[i], 
               games_features['feature_names'].tolist()] ) + noise_games
            
            #Ensure games is not negative
            player_data.loc[i, 'seasons_games_pct'] = np.clip(player_data.loc[i, 'seasons_games_pct'], 0, 1)
    
            #Update the games played by position
            player_data.loc[i, 'games_season'] = player_data.loc[i, 'seasons_games_pct'] * games_per_season
            player_data.loc[i, f'{position}_games'] = player_data.loc[i, 'games_season']
            player_data.loc[i, f'{position}_cumulative_games'] =  safe_get(player_data, i-1, f'{position}_cumulative_games') + player_data.loc[i, f'{position}_games']
      
            #Predict PAV
            player_data.loc[i, 'PAV_per_game_season'] = model_PAV_predict.predict(player_data.loc[[i], 
                value_features['feature_names'].tolist()]) + noise_value
             
            player_data.loc[i, 'PAV_per_game_season'] = np.clip(player_data.loc[i, 'PAV_per_game_season'], 0, 100)
             
            #Calculate total PAV, ensuring that the noise doesnt make the predictions negative
            player_data.loc[i, 'PAV_total']    = player_data.loc[i, 'PAV_per_game_season'] * player_data.loc[i, 'games_season']
             
        # Update last seasons variables
        if i+1 in player_data.index:
            player_data.loc[i+1, 'SEASON'] = player_data.loc[i, 'SEASON'] + 1
            
            #Features - Games total
            player_data.loc[i+1, 'last_seasons_games']       = safe_get(player_data, i, 'games_season')
            player_data.loc[i+1, 'last_two_seasons_games']   = safe_get(player_data, i, 'games_season') + player_data.loc[i, 'last_seasons_games'] 
            player_data.loc[i+1, 'last_three_seasons_games'] = safe_get(player_data, i, 'games_season') + player_data.loc[i, 'last_two_seasons_games'] 
            player_data.loc[i+1, 'last_four_seasons_games']  = safe_get(player_data, i, 'games_season') + player_data.loc[i, 'last_three_seasons_games'] 
            
            #Features - Games cumulative
            player_data.loc[i+1, 'last_seasons_cumulative_games'] = player_data.loc[i, 'last_seasons_cumulative_games'] + player_data.loc[i, 'games_season']
            
            #Features - Games pct
            ####### NOT ENTIRELY ACCURATE !
            player_data.loc[i+1, 'last_seasons_games_pct']        = player_data.loc[i+1, 'last_seasons_games']       / (1 * games_per_season)  
            player_data.loc[i+1, 'last_two_seasons_games_pct']    = player_data.loc[i+1, 'last_two_seasons_games']   / (2 * games_per_season)  
            player_data.loc[i+1, 'last_three_seasons_games_pct']  = player_data.loc[i+1, 'last_three_seasons_games'] / (3 * games_per_season)  
            player_data.loc[i+1, 'last_four_seasons_games_pct']   = player_data.loc[i+1, 'last_four_seasons_games']  / (4 * games_per_season)  

            #Features - Games by position
            player_data.loc[i+1, 'last_seasons_FWD_games']  = player_data.loc[i, 'FWD_games']
            player_data.loc[i+1, 'last_seasons_DEF_games']  = player_data.loc[i, 'DEF_games']
            player_data.loc[i+1, 'last_seasons_MID_games']  = player_data.loc[i, 'MID_games']
            player_data.loc[i+1, 'last_seasons_WING_games'] = player_data.loc[i, 'WING_games']
            player_data.loc[i+1, 'last_seasons_RUCK_games'] = player_data.loc[i, 'RUCK_games']
            
            #Features - Games by position cumulative
            player_data.loc[i+1, 'last_seasons_FWD_cumulative_games']  = player_data.loc[i, 'last_seasons_FWD_cumulative_games']  + player_data.loc[i, 'FWD_games']
            player_data.loc[i+1, 'last_seasons_DEF_cumulative_games']  = player_data.loc[i, 'last_seasons_DEF_cumulative_games']  + player_data.loc[i, 'DEF_games']
            player_data.loc[i+1, 'last_seasons_MID_cumulative_games']  = player_data.loc[i, 'last_seasons_MID_cumulative_games']  + player_data.loc[i, 'MID_games']
            player_data.loc[i+1, 'last_seasons_WING_cumulative_games'] = player_data.loc[i, 'last_seasons_WING_cumulative_games'] + player_data.loc[i, 'WING_games']
            player_data.loc[i+1, 'last_seasons_RUCK_cumulative_games'] = player_data.loc[i, 'last_seasons_RUCK_cumulative_games'] + player_data.loc[i, 'RUCK_games']
            
            #Features - PAV cumulative
            player_data.loc[i+1, 'last_seasons_cumulative_PAV'] = player_data.loc[i, 'last_seasons_cumulative_PAV'] + player_data.loc[i, 'PAV_total']
            
            #Features - PAV 
            player_data.loc[i+1, 'last_seasons_PAV']            = safe_get(player_data, i  , 'PAV_total')
            player_data.loc[i+1, 'last_two_seasons_PAV']        = safe_get(player_data, i  , 'PAV_total') + player_data.loc[i, 'last_seasons_PAV']
            player_data.loc[i+1, 'last_three_seasons_PAV']      = safe_get(player_data, i  , 'PAV_total') + player_data.loc[i, 'last_two_seasons_PAV']
            player_data.loc[i+1, 'last_four_seasons_PAV']       = safe_get(player_data, i  , 'PAV_total') + player_data.loc[i, 'last_three_seasons_PAV']
            
            #Features - PAV per game
            player_data.loc[i+1, 'last_seasons_PAV_per_game']       = np.where(player_data.loc[i+1, 'last_seasons_games']       == 0, 0,  player_data.loc[i+1, 'last_seasons_PAV']       / player_data.loc[i+1, 'last_seasons_games'])
            player_data.loc[i+1, 'last_two_seasons_PAV_per_game']   = np.where(player_data.loc[i+1, 'last_two_seasons_games']   == 0, 0,  player_data.loc[i+1, 'last_two_seasons_PAV']   / player_data.loc[i+1, 'last_two_seasons_games'])
            player_data.loc[i+1, 'last_three_seasons_PAV_per_game'] = np.where(player_data.loc[i+1, 'last_three_seasons_games'] == 0, 0,  player_data.loc[i+1, 'last_three_seasons_PAV'] / player_data.loc[i+1, 'last_three_seasons_games'])
            player_data.loc[i+1, 'last_four_seasons_PAV_per_game']  = np.where(player_data.loc[i+1, 'last_four_seasons_games']  == 0, 0,  player_data.loc[i+1, 'last_four_seasons_PAV']  / player_data.loc[i+1, 'last_four_seasons_games'])
                     
        #Apply corrections for players with less than 4 years under their belt
        player_data.loc[player_data['Years_Played'] == 2, ['last_two_seasons_games_pct', 'last_three_seasons_games_pct', 'last_four_seasons_games_pct']] = \
            player_data.loc[player_data['Years_Played'] == 2, 'last_seasons_games_pct']

        # If Years_Played is 3, then [last_three_seasons_games_pct, last_four_seasons_games_pct] is equal to last_two_seasons_games_pct
        player_data.loc[player_data['Years_Played'] == 3, ['last_three_seasons_games_pct', 'last_four_seasons_games_pct']] = \
            player_data.loc[player_data['Years_Played'] == 3, 'last_two_seasons_games_pct']

        # If Years_Played is 4, then [last_four_seasons_games_pct] is equal to last_three_seasons_games_pct
        player_data.loc[player_data['Years_Played'] == 4, 'last_four_seasons_games_pct'] = player_data.loc[player_data['Years_Played'] == 4, 'last_three_seasons_games_pct']    


     
    # Simulate retirement event
    player_data['retire_probability'] = [model_retire_predict.predict_proba(player_data.loc[[i],
                               retirement_features['feature_names'].tolist()])[0][1] for i in  range(max_age - age + 1)]
    
    # Ensure the player doesn't retire in the first simulated year
    player_data['retire_probability'][0] = 0  
    
    # Function to determine if a player retires based on their probability
    def determine_retirement(row):
        return np.random.rand() < row['retire_probability']
    
    # Use a stochastic method to predict retirement
    player_data['retire_binary'] = player_data.apply(determine_retirement, axis=1).astype(int)
    
    #Use a threshold method to predict retirement
    #pick_data['retire_binary'] = (pick_data['retire_probability'] > 0.7).astype(int)
    
    #Determine the final sequence of retirement
    player_data['retire_final'] = player_data['retire_binary'].cummax()
    
    #Set the sim ID
    player_data['sim_id'] = id
    
    print(f'Player {Player_ID} run number {id}')
    
    return player_data


###Simulate careers



#Create an empty dataframe
simulation_results = pd.DataFrame()

#Simulation parameters
simulate_runs = range(1, 51)

#Run for one player
#ml_data = ml_data[ml_data['Player_ID'] == 'Max_gawn_1991']
#ml_data = ml_data[ml_data['Player_ID'] == 'Mitch_knevitt_2003']


# Simulate careers for different players and concatenate the results
for index, row in ml_data.iterrows():
    for x in simulate_runs:
        simulation_results = pd.concat([simulate_career(
                        row['Player_ID'], row['player_team'], row['SEASON'], row['age'], row['Years_Played'], 
            row['Pick_Number_National'], row['Pick_Number_Rookie'], row['Pick_Number_Pre_Season'], row['Pick_Number_Mid_Season'],
            row['Player_cumulative_trades'], row['Is_Father_Son'],
            row['last_seasons_cumulative_PAV'], row['last_seasons_PAV'], row['last_two_seasons_PAV'], row['last_three_seasons_PAV'], row['last_four_seasons_PAV'], 
            row['last_seasons_PAV_per_game'], row['last_two_seasons_PAV_per_game'], row['last_three_seasons_PAV_per_game'], row['last_four_seasons_PAV_per_game'],
            row['last_seasons_cumulative_games'], row['last_seasons_games'], row['last_two_seasons_games'], row['last_three_seasons_games'], row['last_four_seasons_games'],
            row['last_seasons_games_pct'], row['last_two_seasons_games_pct'], row['last_three_seasons_games_pct'], row['last_four_seasons_games_pct'], 
            row['FWD_games'], row['DEF_games'], row['MID_games'], row['WING_games'], row['RUCK_games'],
            row['FWD_cumulative_games'], row['DEF_cumulative_games'], row['MID_cumulative_games'], row['WING_cumulative_games'], row['RUCK_cumulative_games'],
            row['last_seasons_FWD_games'], row['last_seasons_DEF_games'], row['last_seasons_MID_games'], row['last_seasons_WING_games'], row['last_seasons_RUCK_games'],
            row['last_seasons_FWD_cumulative_games'], row['last_seasons_DEF_cumulative_games'], row['last_seasons_MID_cumulative_games'], row['last_seasons_WING_cumulative_games'], row['last_seasons_RUCK_cumulative_games'],
            row['seasons_games_pct'], row['games_season'], row['PAV_per_game_season'], row['PAV_Total'], x), simulation_results], ignore_index=True)

#Save to CSV
#simulation_results.to_csv('results/career_simulation_results_player.csv', index = False)
    
#Import CSV
#simulation_results = pd.read_csv("results/career_simulation_results_player.csv")

#Remove columns no longer needed
simulation_results = simulation_results[['Player_Team', 'Player_ID', 'age', 'Position', 'sim_id', 'SEASON', 'PAV_total', 'retire_final']]
        
#Filter out season 2023
simulation_results = simulation_results[simulation_results['SEASON'] > 2023]

# Sorting the data by Player_ID and SEASON to ensure the order is correct for rolling calculations
simulation_results = simulation_results.sort_values(by=["Player_ID", "sim_id", "SEASON"])

#For each player, calculate value over next 1 to 10 years

# To calculate future sums, we will shift the data backwards by the corresponding period minus one
years_to_sum = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for years in years_to_sum:
    simulation_results[f'PAV_{years}_years'] = (
        simulation_results.groupby(['Player_ID', 'sim_id'], group_keys=False)  # Use group_keys=False to avoid cross-group contamination
        .apply(lambda x: x['PAV_total'].rolling(window=years, min_periods=1).sum().shift(-years + 1))
    )
    
#Count the number of years remaining for each player for each simulation
expected_years_until_retire = simulation_results[simulation_results['retire_final'] == 0].groupby(['Player_ID', 'sim_id']).size().reset_index(name='Exp_Years_Retire')
   
#Filter out for season 2024 only
simulation_results = simulation_results[simulation_results['SEASON'] == 2024]

#Join expected years until retire
simulation_results = pd.merge(simulation_results,
                              expected_years_until_retire,
                              how = 'left',
                              on = ['Player_ID', 'sim_id'])

#Fill NAs with zeros
simulation_results['Exp_Years_Retire'] = simulation_results['Exp_Years_Retire'].fillna(0)

#Function to detemine expected PAV until retirement
def calculate_expected_pav(row):
    years_until_retire = row['Exp_Years_Retire']
    if years_until_retire == 0:
        return 0
    elif years_until_retire > 10:
        return row['PAV_10_years']
    else:
        # Retrieve the column name dynamically based on the value of Expected_Years_Until_Retire
        column_name = f"PAV_{int(years_until_retire)}_years"
        return row[column_name]

# Apply the function to each row in the DataFrame
simulation_results['Exp_PAV_Retire'] = simulation_results.apply(calculate_expected_pav, axis=1)

# Define the columns for which we need to calculate the median
median_columns = [f'PAV_{i}_years' for i in range(1,11)] + ['Exp_Years_Retire', 'Exp_PAV_Retire']

# Group by 'Player_ID' and calculate the median for the specified columns
simulation_results_summary = simulation_results.groupby(['Player_Team', 'Player_ID', 'age', 'Position'])[median_columns].median().reset_index()

#Save to csv
simulation_results_summary.to_csv('results/career_simulation_results_player_summary.csv', index = False)






