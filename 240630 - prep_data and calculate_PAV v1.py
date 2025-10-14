# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:12:43 2024

@author: Birch Matthew
"""


#%%

### Prepare Champion Data (select columns, map on player data, data checks)

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import pandas as pd
import yaml

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Prepare Champion Data")

#Read data
stats_champion_df = pd.read_csv('AFL Player Stats by Round by Season.csv')
stats_champion_df_columns = pd.read_excel('AFL Player Stats by Round by Season_Columns.xlsx', sheet_name = "Data")
team_name_mapping_df = pd.read_excel('Team Name Mapping.xlsx', sheet_name = "Data")
team_list_df = pd.read_csv('team_list_df.csv', encoding='ISO-8859-1')
debut_df = pd.read_csv('debut_df.csv', encoding='utf-8')
#Import data that has Father-Son flags
picks_df = pd.read_csv('picks_with_dob_df.csv', encoding='utf-8') 

#Change column names to full description
stats_champion_df.columns = stats_champion_df_columns['Column_Long']

#Change GFC to GEE in the SQUAD column
stats_champion_df['SQUAD'].replace('GFC', 'GEE', inplace = True)

#Calculate free kick differential
stats_champion_df["Free Kick Differential"] = stats_champion_df["Frees For"] - stats_champion_df["Frees Against"]

#Calculate effective disposals, kicks and handballs
stats_champion_df['Effective Disposals'] = stats_champion_df['Disposals'] * stats_champion_df['Disposal Effiecency'] / 100
stats_champion_df['Effective Kicks']     = stats_champion_df['Kicks']     * stats_champion_df['Kick Efficiency'] / 100
stats_champion_df['Effective Handballs'] = stats_champion_df['Handballs'] * stats_champion_df['Handball Efficiency'] / 100

#Keep selected columns
with open('columns_to_keep_champion_v2.yaml', 'r') as file:
    columns_to_keep = yaml.safe_load(file)

# Flatten the dictionary into a single list of column names
columns_to_keep = [column for category in columns_to_keep.values() for column in category]

# Filter the DataFrame to keep only the columns in columns_to_keep
stats_champion_df = stats_champion_df[columns_to_keep]

# Split the 'ATHLETE' column at the period to get first inial and last name
split_df = stats_champion_df['ATHLETE'].str.split('\.| ', expand=True)
n_columns = 3
split_df = split_df.iloc[:, :n_columns]  
split_df['player_first_initial'] = split_df[0].str.strip().str[0]
split_df['player_last_name'] = (split_df[1] + ' ' + split_df[2]).str.strip()

stats_champion_df['player_first_initial'] = split_df['player_first_initial']
stats_champion_df['player_last_name'] = split_df['player_last_name']
stats_champion_df['player_last_name_lower_alpha_only'] = stats_champion_df['player_last_name'].str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)

#Map on different variants of team name
stats_champion_df = pd.merge(
                     stats_champion_df, 
                     team_name_mapping_df,
                     how = 'left',
                     on = 'SQUAD')

# Specify the columns to check for duplicates
join_cols = ['SEASON', 'player_first_initial', 'player_last_name_lower_alpha_only', 'Team_Name']

# Find duplicates including the first occurrence
duplicates = team_list_df.duplicated(subset=join_cols, keep=False)

# Drop these duplicates
team_list_df = team_list_df[~duplicates]
team_list_df.drop(columns = ['player_last_name'], inplace = True)

stats_champion_df = pd.merge(stats_champion_df, 
                             team_list_df,
                             how = 'left',
                             on = join_cols
                             )

#Remove rows where player name is not joined property (This occurs for about 351 out of 150,000 rows)
stats_champion_df = stats_champion_df.dropna(subset=['player_name'])

#Add a player identifier
stats_champion_df['birth_year'] = stats_champion_df['birth_year'].astype('Int64')
stats_champion_df['Player_ID'] = (stats_champion_df['player_first_name'] + '_' +
                                  stats_champion_df['player_last_name_lower_alpha_only'] + '_' +
                                  stats_champion_df['birth_year'].astype('str')
                                  )

#Clean up the draft pick column using a regular expression to pick up pick number, draft type and year drafted
pattern = r'#?(\d{1,2})?\s*(\D+?)\s*(\d{4})'
stats_champion_df[['Pick_Number', 'Draft_Type', 'Year_Drafted']] = stats_champion_df['drafted'].str.extract(pattern)

### Join the debut dataframe to get the debut year

#Join onto main dataframe
debut_df['Player_ID'] = (debut_df['player_first_name'] + '_' +
                         debut_df['player_last_name_lower_alpha_only'] + '_' +
                         debut_df['Birth_Year'].astype('str')
                         )

stats_champion_df = pd.merge(stats_champion_df,
                             debut_df[['Player_ID', 'Debut_Year']],
                             how = 'left',
                             on = ['Player_ID'])


### Join the picks dataframe to get Father-Son selections

stats_champion_df = pd.merge(stats_champion_df,
                             picks_df[picks_df['Is_Father_Son'] == 1][['Player_ID', 'Is_Father_Son']], 
                             on = ['Player_ID'],
                             how = 'left')
     
stats_champion_df['Is_Father_Son'].fillna(0, inplace = True)
                        
### Data quality checks

#Check which columns have missing values
missing_values = stats_champion_df.isnull().sum()
print(f"Columns with missing values: \n{missing_values[missing_values > 0]}")

# Flag all duplicate rows based on the specified columns
duplicates = stats_champion_df.duplicated(subset=['SEASON', 'ROUND', 'SQUAD', 'player_first_initial', 'player_last_name'], keep=False)

# Filter the DataFrame to only include the duplicate rows
duplicate_rows = stats_champion_df[duplicates]

print(f"Number of duplicate rows: {len(duplicate_rows)}")

#Save to csv
stats_champion_df.to_csv('AFL Player Stats by Round by Season_Prepared.csv',index=False)


#%%

### Process fitzRoy data

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import pandas as pd
import yaml
from datetime import datetime

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Prepare fitzRoy data")

#Read data
stats_fitzroy_df = pd.read_csv('fryzigg_df.csv')

#Keep selected columns
with open('columns_to_keep_fitzroy_v1.yaml', 'r') as file:
    columns_to_keep = yaml.safe_load(file)

# Flatten the dictionary into a single list of column names
columns_to_keep = [column for category in columns_to_keep.values() for column in category]

# Filter the DataFrame to keep only the columns in columns_to_keep
stats_fitzroy_df = stats_fitzroy_df[columns_to_keep]

#Get the players first name initial 
stats_fitzroy_df['player_first_initial'] = stats_fitzroy_df['player_first_name'].str[:1].str.strip().str[0]

#Get the season year from match date
stats_fitzroy_df['match_date'] = pd.to_datetime(stats_fitzroy_df['match_date'], format='%Y-%m-%d')
stats_fitzroy_df['SEASON'] = stats_fitzroy_df['match_date'].dt.year

#Convert match_round text round to a number (i.e. there are values that say "Grand Final")
stats_fitzroy_df['numerical_round'] = pd.to_numeric(stats_fitzroy_df['match_round'], errors='coerce')

# Find the maximum round number for each season
max_round_per_season = stats_fitzroy_df.groupby('SEASON')['numerical_round'].max().reset_index()
max_round_per_season.rename(columns={'numerical_round': 'max_round'}, inplace=True)

# Merge this info back to the original df to know the max round per row/season
stats_fitzroy_df = stats_fitzroy_df.merge(max_round_per_season, on='SEASON')

#Dictionary that has the finals week number
finals_week_dict = {
    'Qualifying Final' : 1	,
    'Elimination Final' : 1,
    'Finals Week 1' : 1,
    'Semi Final' : 2,
    'Semi Finals' : 2,
    'Preliminary Final' : 3,
    'Preliminary Finals' : 3,
    'Grand Final' : 4
    }

#Function to map the finals round number
def map_finals_round(row):
    if pd.isna(row['numerical_round']):  # It's a finals match
        return row['max_round'] + finals_week_dict.get(row['match_round'], 1)
    else:
        return row['numerical_round']

# Apply this function row-wise
stats_fitzroy_df['ROUND'] = stats_fitzroy_df.apply(map_finals_round, axis=1)

# Drop the intermediate 'max_round' and 'numerical_round' columns
stats_fitzroy_df.drop(['max_round', 'numerical_round'], axis=1, inplace=True)

#Due to drawn grand final, if the match date is "2/10/2010", then set round = 27
stats_fitzroy_df.loc[stats_fitzroy_df['match_date'] == '2010-10-02', 'ROUND'] = 27

#Get the variant of the last name
stats_fitzroy_df['player_last_name_lower_alpha_only'] = stats_fitzroy_df['player_last_name'].str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)

#Calculate the number of matches that each team played each season
SQUAD_Matches_Played = stats_fitzroy_df.groupby(['SEASON', 'player_team'])['match_date'].nunique().reset_index(name='SQUAD_Matches_Played').dropna()
SQUAD_Matches_Played = SQUAD_Matches_Played.sort_values(by=['player_team', 'SEASON'])
SQUAD_Matches_Played['last_seasons_squad_games'] = SQUAD_Matches_Played.groupby('player_team')['SQUAD_Matches_Played'].shift(1)
SQUAD_Matches_Played['last_two_seasons_squad_games'] = SQUAD_Matches_Played.groupby('player_team')['SQUAD_Matches_Played'].transform(lambda x: x.rolling(window=2, min_periods=1).sum().shift())
SQUAD_Matches_Played['last_three_seasons_squad_games'] = SQUAD_Matches_Played.groupby('player_team')['SQUAD_Matches_Played'].transform(lambda x: x.rolling(window=3, min_periods=1).sum().shift())
SQUAD_Matches_Played['last_four_seasons_squad_games'] = SQUAD_Matches_Played.groupby('player_team')['SQUAD_Matches_Played'].transform(lambda x: x.rolling(window=4, min_periods=1).sum().shift())
SQUAD_Matches_Played = SQUAD_Matches_Played[SQUAD_Matches_Played['SEASON'] >= 2004]

### Data quality checks

# Identify columns with any missing values
columns_with_missing_values = stats_fitzroy_df.columns[stats_fitzroy_df.isnull().any()].tolist()

print(f"Number of columns with missing values: {len(columns_with_missing_values)}")

# Drop these columns
stats_fitzroy_df = stats_fitzroy_df.drop(columns=columns_with_missing_values)

# Step 3: Report dropped columns
print(f"Dropped columns that have missing values: {columns_with_missing_values}")

#Check duplicate rows
duplicate_rows = stats_fitzroy_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

#Save to csv
stats_fitzroy_df.to_csv('fryzigg_df_Prepared.csv',index=False)
SQUAD_Matches_Played.to_csv('squad_matches_df.csv',index=False)


#%%

### Join Chamption Data onto fitzRoy data

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import pandas as pd
import yaml

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Joining Chamption Data with fitzRoy")

#Read data
stats_champion_df = pd.read_csv('AFL Player Stats by Round by Season_Prepared.csv')
stats_fitzroy_df = pd.read_csv('fryzigg_df_Prepared.csv')

# Mark all duplicates, including the first occurrence, in the specified columns
duplicates = stats_fitzroy_df.duplicated(subset=['player_team', 'player_first_name', 'player_last_name_lower_alpha_only', 'ROUND', 'SEASON'], keep=False)

# Keep rows that are not duplicates
stats_fitzroy_df = stats_fitzroy_df[~duplicates]

#Get the first 3 letters of players first name
stats_fitzroy_df['player_first_name_3_letters'] = stats_fitzroy_df['player_first_name'].str[:3]
stats_fitzroy_df = stats_fitzroy_df.drop(columns=['player_first_name', 'player_last_name'])

#Join the fitzroy data on to Champion
stats_champion_df = pd.merge(stats_champion_df, 
                             stats_fitzroy_df, 
                             how = 'inner',
                             on = ['player_team', 'player_first_initial', 'player_first_name_3_letters', 'player_last_name_lower_alpha_only', 'ROUND', 'SEASON'])

### Data quality checks

# Identify columns with any missing values

# Calculate the number of missing values for each column
missing_values = stats_champion_df.isnull().sum()

# Filter to only show columns with missing values
missing_values = missing_values[missing_values > 0]

print(f"Columns with missing values:\n{missing_values}")

#Check duplicate rows
duplicate_rows = stats_champion_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

#Save to csv
stats_champion_df.to_csv('AFL Player Stats by Round by Season_Joined.csv',index=False)



#%%

### Create the features for the PAV model

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import pandas as pd
from scipy.stats import mode
import yaml
import re

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Creating features for the PAV model")

#Read data
stats_champion_df = pd.read_csv('AFL Player Stats by Round by Season_Joined.csv')

#Config file
with open('columns_to_keep_fitzRoy_v1.yaml', 'r') as file:
    columns_to_keep_fitzroy = yaml.safe_load(file)

with open('columns_to_keep_champion_v2.yaml', 'r') as file:
    columns_to_keep_champion = yaml.safe_load(file)    

###Player table

# Group by player, season and team and apply the sumproduct function for disposals
groupby = ["SEASON", 
           "SQUAD", 'player_team',
           "ATHLETE", 'player_first_name', 'player_last_name', 'Player_ID',
           'date_of_birth', 'birth_year', 'age', 
           'height', 'weight',
           'Pick_Number', 'Draft_Type', 'Is_Father_Son', 'Year_Drafted', 'Debut_Year',
           'games_prior', 'goals_prior', 'votes_prior', 'games_season', 'goals_season', 'votes_season'
           ]

# Sum all the other columns
[columns_to_keep_champion['pav_stats'].remove(item) for item in ['Disposal Effiecency', 'Kick Efficiency', 'Handball Efficiency'] if item in columns_to_keep_champion['pav_stats']]
columns_to_sum = columns_to_keep_champion['pav_stats'] + columns_to_keep_fitzroy['pav_stats']

# Create an aggregation dictionary based on the columns to sum
agg_dict = {col: 'sum' for col in columns_to_sum}

# Get the most common position that a player was in throughout the season
agg_dict['POS'] = lambda x: x.mode()[0]

#Aggregate the column
player_df = stats_champion_df.groupby(groupby, dropna=False).agg(agg_dict).reset_index()

#Save to csv
player_df.to_csv('player_df.csv',index=False)

###Team table

#Aggregate Inside 50s, goals and behinds for both team and conceeded
team_df = pd.merge(stats_champion_df.groupby(['SEASON', 'SQUAD'])[['Inside 50s', 'Goals', 'Behinds', 'Ruck Contests', 'Hit Outs to Advantage']].sum().reset_index(),
                   stats_champion_df.groupby(['SEASON', 'OPP'])[['Inside 50s', 'Goals', 'Behinds']].sum().reset_index(),
                   how = 'inner',
                   left_on = ['SEASON', 'SQUAD'],
                   right_on = ['SEASON', 'OPP'])

team_df.drop(columns = 'OPP', inplace = True)

#Rename columns
columns_to_rename = {
    'Inside 50s_x': 'Inside 50s',
    'Goals_x': 'Goals',
    'Behinds_x': 'Behinds',
    'Inside 50s_y': 'Inside 50s Conceeded',
    'Goals_y': 'Goals Conceeded',
    'Behinds_y': 'Behinds Conceeded'
}

# Rename the columns
team_df.rename(columns=columns_to_rename, inplace=True)

#Calculate total points and points conceeded (Note: Rushed behinds are missing, but can be considered insignificant)
team_df['Points'] = (6 * team_df['Goals'] + team_df['Behinds'])
team_df['Points Conceeded'] = (6 * team_df['Goals Conceeded'] + team_df['Behinds Conceeded'])
                   
#Calculate raw line scores
team_df['team_offence_score_raw']  = team_df['Points'] / team_df['Inside 50s']
team_df['team_midfield_score_raw'] = team_df['Inside 50s'] / team_df['Inside 50s Conceeded']
team_df['team_defence_score_raw']  = team_df['Points Conceeded'] / team_df['Inside 50s Conceeded']
team_df['team_ruck_score_raw']     = team_df['Hit Outs to Advantage'] / team_df['Ruck Contests']


#Calculate ratios to scale the average to 1
team_df['season_ave_offence_ratio']  = team_df.groupby('SEASON')['team_offence_score_raw'].transform('mean')
team_df['season_ave_midfield_ratio'] = team_df.groupby('SEASON')['team_midfield_score_raw'].transform('mean')
team_df['season_ave_defence_ratio']  = team_df.groupby('SEASON')['team_defence_score_raw'].transform('mean')
team_df['season_ave_ruck_ratio']     = team_df.groupby('SEASON')['team_ruck_score_raw'].transform('mean')

#Calculate final scaled scores
team_df['Team_FWD_Score'] = team_df['team_offence_score_raw'] / team_df['season_ave_offence_ratio']
team_df['Team_MID_Score'] = team_df['team_midfield_score_raw'] / team_df['season_ave_midfield_ratio']
team_df['DN'] = team_df['team_defence_score_raw'] / team_df['season_ave_defence_ratio']
team_df['Team_DEF_Score'] = (100 * (((2 * team_df['DN']) - (team_df['DN']**2)) / (2 * team_df['DN']))) * 2 / 100
team_df['Team_RUCK_Score'] = team_df['team_ruck_score_raw'] / team_df['season_ave_ruck_ratio']

#Save to csv
team_df.to_csv('team_df.csv',index=False)


#%%

### Run the PAV prediction

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value'
os.chdir(file_path) 

#Import packages
import pandas as pd
import numpy as np
import yaml

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Load data
player_df = pd.read_csv('data/player_df.csv', encoding='utf-8')
team_df = pd.read_csv('data/team_df.csv', encoding='utf-8')

#Read the model weights
def read_yaml_file(filename):
    with open(filename, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

#The function to calculate the score component based on weights and player data.
def calculate_score_component(data, weights):   
    score = np.zeros(len(data))
    for column, weight in weights.items():
        score += data[column] * weight
    return score

#This function calculates the score by applying the score components across different roles.
def calculate_score(player_table, params):
    return pd.DataFrame(
                {
                    "SEASON": player_table["SEASON"],
                    "SQUAD": player_table["SQUAD"],
                    "POS": player_table["POS"],
                    "Player_ID": player_table["Player_ID"],     
                    **{
                        f"{role}_Score": calculate_score_component(
                            player_table, weights
                        )
                        for role, weights in params.items()
                    },
                }
            )
    

#Normalize the scores to team totals for each season.
def normalise_score(score_table):
    score_cols = [
        c for c in score_table.columns if c not in ["SEASON", "SQUAD", "Player_ID", 'POS']
    ]
    teams_total_score_season = (
        score_table.groupby(["SEASON", "SQUAD"])[score_cols].sum().add_prefix("Team_Total_")
    )
    score_table = score_table.merge(
        teams_total_score_season, on=["SEASON", "SQUAD"], how="left"
    )
    
    for col in score_cols:
        score_table[f"Normalised_{col}"] = (
            score_table[col] / score_table[f"Team_Total_{col}"]
        )
    return score_table


#Merge the normalized score table with team information.
def append_team_table(normalised_score, team_table):
    return pd.merge(
        normalised_score,
        team_table[['SEASON', 'SQUAD', 'Team_FWD_Score', 'Team_MID_Score', 'Team_DEF_Score', 'Team_RUCK_Score']],
        on=["SEASON", "SQUAD"],
        how="left"
    )

#Calculate the final PAV scores for different roles.
def calc_pav(score_norm):
    score_norm['PAV_Gen Def'] = score_norm['Normalised_Gen Def_Weights_Score'] * score_norm['Team_DEF_Score'] * 100
    score_norm['PAV_Key Def'] = score_norm['Normalised_Key Def_Weights_Score'] * score_norm['Team_DEF_Score'] * 100
    
    score_norm['PAV_Gen Fwd'] = score_norm['Normalised_Gen Fwd_Weights_Score'] * score_norm['Team_FWD_Score'] * 100
    score_norm['PAV_Key Fwd'] = score_norm['Normalised_Key Fwd_Weights_Score'] * score_norm['Team_FWD_Score'] * 100
    
    score_norm['PAV_Mid']     = score_norm['Normalised_Mid_Weights_Score']     * score_norm['Team_MID_Score'] * 100
    score_norm['PAV_Mid-Fwd'] = score_norm['Normalised_Mid-Fwd_Weights_Score'] * score_norm['Team_MID_Score'] * 100
    score_norm['PAV_Wing']    = score_norm['Normalised_Wing_Weights_Score']    * score_norm['Team_MID_Score'] * 100
    score_norm['PAV_Ruck']    = score_norm['Normalised_Ruck_Weights_Score']    * score_norm['Team_MID_Score'] * 100
    
    score_norm["PAV_Total"] = (
          score_norm["PAV_Gen Def"].fillna(0)
        + score_norm["PAV_Key Def"].fillna(0)
        + score_norm["PAV_Gen Fwd"].fillna(0)
        + score_norm["PAV_Key Fwd"].fillna(0)
        + score_norm["PAV_Mid"].fillna(0)
        + score_norm["PAV_Mid-Fwd"].fillna(0)
        + score_norm["PAV_Wing"].fillna(0)
        + score_norm["PAV_Ruck"].fillna(0)
    )
    return score_norm

# Assuming player_table and team_table are your input data frames and params is your loaded parameters
default_params = read_yaml_file('data/model_weights_v6.yml')
score = calculate_score(player_df, default_params)

#Get all the columns that have scores
score_columns = [i+'_Weights_Score' for i in list(score['POS'].unique())]

#Set all the values to zero if the player did not play in that position
for index, row in score.iterrows():
    for col in score_columns:
        if col != row['POS'] + '_Weights_Score':
            score.loc[index, col] = 0

normalised_score = normalise_score(score)
score_with_team_info = append_team_table(normalised_score, team_df)
predicted_pav_df = calc_pav(score_with_team_info)

#Save to csv
predicted_pav_df.to_csv('results/predicted_pav_df.csv',index=False)



