# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:42:16 2024

@author: Birch Matthew
"""

#%%

### get_player_draft_picks from draftguru

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import re
from urllib.request import urlopen
import pandas as pd
import time
import random
from datetime import datetime

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Getting pick data from https://www.draftguru.com.au/years/<year_number>")

#Use beautiful soup to get the data from this website:
#https://www.draftguru.com.au/years/{year_number}
#e.g. https://www.draftguru.com.au/years/2023

#Specify the url for draft picks
url_template = "https://www.draftguru.com.au/years/{year_number}"
min_year = 1981
max_year = 2023
                             
# Regular expression patterns for year, club and player
draft_type_pattern   = r'<td class="draft">(.*?)</td>'
pick_number_pattern  = r'<td class="number">(.*?)</td>'
club_name_pattern    = r'<td class="club"><a href="/clubs/[^"]+">(.*?)</a></td>'
signing_pattern      = r'<td class="category">(Father-Son Selection)'
player_name_pattern  = r'<td class="player".*?><a href="/players/[^"]+">(.*?)</a></td>'
age_at_draft_pattern = r'<td class="stats">(\d+)<span class="units">yr</span></td>'

#Create an empty dataframe to store all the results
picks_df = pd.DataFrame()

#Iterate over every year to get draft pick data
for year_number in range (min_year, max_year + 1):
    
    #Check if the year data is already available
    if 'Year' in picks_df.columns:
        if year_number in picks_df['Year'].values:
            #If it is, skip to the next year
            print(f"{year_number} already available")
            continue

    # List to hold each row's data
    players_info = []

    # Temporary storage for the current record being processed
    current_player_info = {}

    #Set the URL for the specified pick
    print(f"Scraping for year {year_number}")
    url = url_template.format(year_number = year_number)
    
    #Try to scrape the data, ensuring some buffer time between scrapes
    try:
        #Random delay of between 1 and 5 seconds
        time.sleep(random.uniform(1, 5))
        response = urlopen(url)
        print(f"First scrape attempt successful")
        
    #Otherwise, wait longer then try again
    except Exception as e:
        time.sleep(random.uniform(20, 30))
        response = urlopen(url)
        print(f"Second scrape attempt successful")
        
    #Otherwise, wait even longer then try again
    except Exception as e:
        time.sleep(random.uniform(30, 60))
        response = urlopen(url)
        print(f"Third scrape attempt successful")
    
    #Decode the data and split into lines
    long_txt = response.read().decode()
    data = long_txt.split('\n')

    #Iterate over each line in the data     
    for line in data:
        #Determine if the row has useful information
        draft_type_match = re.search(draft_type_pattern, line)
        pick_number_match = re.search(pick_number_pattern, line)
        club_name_match = re.search(club_name_pattern, line)
        signing_match = re.search(signing_pattern, line)
        player_name_match = re.search(player_name_pattern, line)
        age_at_draft_match = re.search(age_at_draft_pattern, line)
        
        if draft_type_match:
            current_player_info['Draft Type'] = draft_type_match.group(1)
        if pick_number_match:
            current_player_info['Pick Number'] = pick_number_match.group(1)
        if club_name_match:
            current_player_info['Club Name'] = club_name_match.group(1)
        if signing_match:
            current_player_info['Signing'] = signing_match.group(1)
        if player_name_match:
            current_player_info['Player Name'] = player_name_match.group(1).replace("&nbsp;", " ")
            # As soon as a player name is found, add the current player to the list
            # and prepare for the next player's information
            players_info.append(current_player_info)
            current_player_info = {}
            # Reset current_player_info but carry over repetitive data for the next player
            current_player_info['Draft Type'] = players_info[-1].get('Draft Type', '')
        if age_at_draft_match:
            # Update the last added player info with age, as player is already added to the list
            players_info[-1]['Age at Draft'] = age_at_draft_match.group(1)
    
    # Create a DataFrame
    year_df = pd.DataFrame(players_info)

    #Add the year column
    year_df['Year'] = year_number
    
    #Append to master data
    picks_df = pd.concat([picks_df, year_df], ignore_index=True)

# Ensure "Year" is converted to numeric, just in case it's not.
picks_df["Year"] = pd.to_numeric(picks_df["Year"], errors='coerce')

# Convert 'Age at Draft' to numeric, coercing errors. This will replace non-numeric values with NaN.
picks_df['Age at Draft'] = pd.to_numeric(picks_df['Age at Draft'], errors='coerce')
  
## Calculate approx player year of birth
picks_df["Approx_Birth_Year"] = picks_df["Year"] - picks_df['Age at Draft']

# Ensure 'Birth_Year' is an integer to avoid decimals
picks_df['Approx_Birth_Year'] = pd.to_numeric(picks_df['Approx_Birth_Year'], errors='coerce').astype('Int64')

# Flag whether the pick is a father-son selection
picks_df['Is_Father_Son'] = picks_df['Signing'].apply(lambda x: 1 if 'Father-Son Selection' in str(x) else 0)

#Save to csv   
picks_df.to_csv('picks_df.csv',index=False)


#%%

### get_player_dobs from draftguru

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import re
from urllib.request import urlopen
import pandas as pd
import time
import random
from datetime import datetime

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Getting dob data from https://www.draftguru.com.au/birth-months/<month_number>")

#Specify the url for date of births
url_template = "https://www.draftguru.com.au/birth-months/{month_number}"

# Regular expression patterns for year, club and player
player_name_pattern = r'<td class="rowname"><a href="/players/[^"]+">(.*?)</a></td>'
dob_pattern = r'<td class="info" data-sorttable_customkey="(\d{4}-\d{2}-\d{2})">\d{4}-\d{2}-\d{2}</td>'

#Create an empty dataframe to store all the results
dob_df = pd.DataFrame()

#Iterate over every month to get date of births
for month_number in range (1, 13):
    
    #Check if the month data is already available
    if 'Month' in dob_df.columns:
        if month_number in dob_df['Month'].values:
            #If it is, skip to the next year
            print(f"{month_number} already available")
            continue

    # List to hold each row's data
    players_info = []

    # Temporary storage for the current record being processed
    current_player_info = {}

    #Set the URL for the specified month
    print(f"Scraping for month {month_number}")
    url = url_template.format(month_number = month_number)
    
    #Try to scrape the data, ensuring some buffer time between scrapes
    try:
        #Random delay of between 1 and 5 seconds
        time.sleep(random.uniform(1, 5))
        response = urlopen(url)
        print(f"First scrape attempt successful")
        
    #Otherwise, wait longer then try again
    except Exception as e:
        time.sleep(random.uniform(20, 30))
        response = urlopen(url)
        print(f"Second scrape attempt successful")
        
    #Otherwise, wait even longer then try again
    except Exception as e:
        time.sleep(random.uniform(30, 60))
        response = urlopen(url)
        print(f"Third scrape attempt successful")
    
    #Decode the data and split into lines
    long_txt = response.read().decode()
    data = long_txt.split('\n')

    #Iterate over each line in the data     
    for line in data:
        #Determine if the row has useful information
        player_name_match = re.search(player_name_pattern, line)
        dob_match = re.search(dob_pattern, line)
        
        #If there is a match, then store the data
        if player_name_match:
            current_player_info['Player Name'] = player_name_match.group(1).replace("&nbsp;", " ")
        if dob_match:
            current_player_info['DOB'] = dob_match.group(1)
            #Once DOB is found, add it to the main list, and reset the dictionary
            players_info.append(current_player_info)            
            current_player_info = {}
    
    # Create a DataFrame
    month_df = pd.DataFrame(players_info)
    
    #Add the year column
    month_df['Month'] = month_number
    
    #Append to master data
    dob_df = pd.concat([dob_df, month_df], ignore_index=True)

# Convert the DOB to datetime and extract the year
dob_df['DOB'] = pd.to_datetime(dob_df['DOB'], format='%Y-%m-%d')
dob_df['Year'] = dob_df['DOB'].dt.year

#Save to csv
dob_df.to_csv('dob_df.csv',index=False)


#%%

### Join DOBs onto picks data

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import pandas as pd

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Getting joining picks data onto dob data")

#Read data
picks_df = pd.read_csv('picks_df.csv')
dob_df = pd.read_csv('dob_df.csv')

# Create a function to find the closest birth year match for each player within 1 year difference
def find_closest_dob_within_one_year(row, dob_df):
    # Skip the row if 'Approx_Birth_Year' is NaN
    if pd.isna(row['Approx_Birth_Year']):
        return pd.NaT
    
    player_name = row['Player Name']
    birth_year = row['Approx_Birth_Year']
    # Filter dob_df for rows with the same player name
    filtered_dob = dob_df[dob_df['Player Name'] == player_name]
    # Compute the difference in years
    year_diff = (filtered_dob['Year'] - birth_year).abs()
    # Find the row with the closest year that's within 1 year difference
    closest_dob = filtered_dob.loc[year_diff.idxmin()] if not year_diff.empty else None
    if closest_dob is not None and year_diff.min() <= 1:
        # Return the date of birth if a match is found
        return closest_dob['DOB']
    else:
        # Return NaN if no match is found or the difference is more than 1 year
        return pd.NaT

# Apply the function to the picks_df to find the closest DOB for each row
picks_df['DOB'] = picks_df.apply(lambda row: find_closest_dob_within_one_year(row, dob_df), axis=1)

# Assuming picks_df['DOB'] has been previously converted to datetime and may contain NaT values
picks_df['DOB'] = pd.to_datetime(picks_df['DOB'], errors='coerce')  # just to ensure it's in datetime format

#Add in the acutal birth year column
picks_df['Birth_Year'] = picks_df['DOB'].dt.year

#Remove Approx Birth Year column
picks_df = picks_df.drop(columns = ['Approx_Birth_Year', 'Signing'], axis=1)

#Remove players without a DOB
picks_df = picks_df.dropna(subset=['DOB'])

# Convert 'Birth_Year' to a nullable integer type, which will convert NaNs to pandas NA values
picks_df['Birth_Year'] = pd.to_numeric(picks_df['Birth_Year'], errors='coerce').astype('Int64')

# Create a unique player identifier using player name and birth year
picks_df[['player_first_name', 'player_last_name']] = picks_df['Player Name'].str.split(' ', n=1, expand=True)
picks_df['player_last_name_lower_alpha_only'] = picks_df['player_last_name'].str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)

picks_df['Player_ID'] = (picks_df['player_first_name'] + '_' +
                         picks_df['player_last_name_lower_alpha_only'] + '_' +
                         picks_df['Birth_Year'].astype('str')
                          )

# Set debut club
picks_df['First Club'] = picks_df.sort_values('Year').groupby('Player_ID')['Club Name'].transform('first')

##Potential furuthre updates
#PLayer who was picked up in pre-draft then traded in the same year might have the wrong first club
#Could add in data from another source e.g. https://afltables.com/afl/stats/biglists/bg10.txt

#Save to csv
picks_df.to_csv('picks_with_dob_df.csv',index=False)



#%%

### get_player_debut dates from afl tables

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import re
from urllib.request import urlopen
import pandas as pd
import time
import random
from datetime import datetime

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Getting debut data from https://afltables.com/afl/stats/biglists/bg10.txt")

#Specify the url for date of births
url = "https://afltables.com/afl/stats/biglists/bg10.txt"

# Regular expression patterns
pattern = r"""
    \d+\.        # Match the starting number and period
    \s+          # Match any whitespace following the period
    ([A-Za-z\s\-]+?)   # Matching the player's name which might include hyphens or apostrophes.
    \s+          # Match the whitespace after the name
    (\d{1,2}-\w{3}-\d{4}) # Capture the date of birth
    \s+          # Match the whitespace after the date of birth
    (R\d+|SF|PF|GF|EF)       # Capture the debut round (e.g., "R8")
    \s+          # Match the whitespace after the debut round
    (\w{2})      # Capture the debut team acronym (e.g., "CW")
    \s+v\s+\w{2} # Match 'v' and the opposing team acronym
    \s+          # Match the whitespace after the versus
    (\d{1,2}-\w{3}-\d{4}) # Capture the debut date
"""


# List to store each player's information as a dictionary
players_data = []

#Try to scrape the data, ensuring some buffer time between scrapes
try:
    #Random delay of between 1 and 5 seconds
    time.sleep(random.uniform(1, 5))
    response = urlopen(url)
    print(f"First scrape attempt successful")
    
#Otherwise, wait longer then try again
except Exception as e:
    time.sleep(random.uniform(20, 30))
    response = urlopen(url)
    print(f"Second scrape attempt successful")
    
#Otherwise, wait even longer then try again
except Exception as e:
    time.sleep(random.uniform(30, 60))
    response = urlopen(url)
    print(f"Third scrape attempt successful")

#Decode the data and split into lines
long_txt = response.read().decode()
data = long_txt.split('\n')

# Compile the regular expression pattern with VERBOSE flag for multi-line and commented pattern
compiled_pattern = re.compile(pattern, re.VERBOSE)

# Process each line to extract the information and add it to the list
for line in data:
    match = compiled_pattern.search(line)
    if match:
        player_info = {
            'Player Name': match.group(1).strip(),
            'DOB': match.group(2),
            'Debut Round': match.group(3),
            'Debut Team': match.group(4),
            'Debut Date': match.group(5)
        }
        players_data.append(player_info)

# Create DataFrame from the list of dictionaries
debut_df = pd.DataFrame(players_data)

# Convert the DOB and debut date to datetime and extract the year
debut_df['DOB'] = pd.to_datetime(debut_df['DOB'], format='%d-%b-%Y')
debut_df['Birth_Year'] = debut_df['DOB'].dt.year

debut_df['Debut Date'] = pd.to_datetime(debut_df['Debut Date'], format='%d-%b-%Y')
debut_df['Debut_Year'] = debut_df['Debut Date'].dt.year

# Define a function to split the name and return the first initial and last name
def split_name(name):
    parts = name.split()
    if len(parts) > 1:  # Check if the name consists of at least a first name and last name
        first_initial = parts[0][0]  # First character of the first name
        last_name = parts[-1]  # Last name
        return first_initial, last_name
    else:
        return '', name  # If there's only one part, return it as the last name

#Create variations of player name
debut_df[['player_first_name', 'player_last_name']] = debut_df['Player Name'].str.split(' ', n=1, expand=True)
debut_df['player_first_initial'] = debut_df['player_first_name'].str[:3]
debut_df['player_first_name_3_letters'] = debut_df['player_first_name'].str[:1]
debut_df['player_last_name_lower_alpha_only'] = debut_df['player_last_name'].str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)

#Drop duplicate rows
debut_df = debut_df.drop_duplicates(subset=['Player Name', 'Birth_Year'])

#Save to csv
debut_df.to_csv('debut_df.csv',index=False)




#%%

### get_team_list for draft guru, which has for every year and every team, the players and their details


#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import pandas as pd
import time
import random
from datetime import datetime
import requests
from bs4 import BeautifulSoup

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info(f"Getting dob data from https://www.draftguru.com.au/lists/<year_number>/<team_name>")

#Specify the url for date of births
url_template = "https://www.draftguru.com.au/lists/{year_number}/{team_name}"

#Create an empty master dataframe
team_list_df = pd.DataFrame()

#Specify the team names
team_names = ['adelaide','brisbane','carlton','collingwood','essendon','fitzroy','fremantle',
'geelong','gold-coast','greater-western-sydney','hawthorn','melbourne','north-melbourne',
'port-adelaide','richmond','st-kilda','sydney','west-coast','western-bulldogs']

for year_number in range(1990, 2024):
    for team_name in team_names:
        
        #Check if the team data in the year is already available
        if ('SEASON' in team_list_df.columns) and ('SEASON' in team_list_df.columns):
            has_values = not team_list_df[(team_list_df['Team_Name'] == team_name) & (team_list_df['SEASON'] == year_number)].empty
            
            if has_values:
                #If it is, skip to the next year and team
                print(f"Data for {team_name} in {year_number} is already available")
                continue
        
        # The URL of the page you want to scrape
        print(f"Scraping for team {team_name} in year {year_number}")
        url = url_template.format(team_name = team_name, year_number = year_number)
              
        #Try to scrape the data, ensuring some buffer time between scrapes
        try:
            #Random delay of between 1 and 5 seconds
            time.sleep(random.uniform(1, 5))
            # Send an HTTP request to the URL
            response = requests.get(url)
            print(f"First scrape attempt successful")
            
        #Otherwise, wait longer then try again
        except Exception as e:
            time.sleep(random.uniform(20, 30))
            # Send an HTTP request to the URL
            response = requests.get(url)
            print(f"Second scrape attempt successful")
            
        #Otherwise, wait even longer then try again
        except Exception as e:
            time.sleep(random.uniform(30, 60))
            # Send an HTTP request to the URL
            response = requests.get(url)
            print(f"Third scrape attempt successful")
        
        # Check if the request was successful
        if response.ok:
            # Get the content of the response
            html_content = response.text
            
            # Create a BeautifulSoup object and specify the parser
            soup = BeautifulSoup(html_content, 'html.parser')
        else:
            print(f"Failed to retrieve content: {response.status_code}")
                  
        players_data = []
        
        # Assuming each player's data is in a separate 'tr' tag
        for tr in soup.find_all('tr'):
            # Initial counts for 'games', 'goals', and 'votes' to handle first and second appearances
            games_count = goals_count = votes_count = 0
        
            player_data = {}
            for td in tr.find_all('td'):
                class_name = td.get('class')[0] if td.get('class') else ''
        
                # Handle repeating fields by counting occurrences
                if class_name in ['games', 'goals', 'votes']:
                    count_var_name = f'{class_name}_count'
                    locals()[count_var_name] += 1
                    field_name = f'{class_name}_prior' if locals()[count_var_name] == 1 else f'{class_name}_season'
                else:
                    field_name = class_name
        
                # Mapping class names to dictionary keys and extracting text
                if field_name == 'name':
                    player_data['player_name'] = td.get_text(strip=True).replace('&nbsp;', ' ')
                elif field_name == 'dob':
                    player_data['date_of_birth'] = td.span.get_text(strip=True) if td.span else ''
                elif field_name == 'age':
                    player_data['age'] = td.get_text(strip=True).replace('yr', '')
                elif field_name in ['height', 'weight', 'games_prior', 'goals_prior', 'votes_prior', 'games_season', 'goals_season', 'votes_season']:
                    player_data[field_name] = td.get_text(strip=True).replace('cm', '').replace('kg', '')
                elif field_name == 'movement':
                    player_data['drafted'] = td.get_text(strip=True)
        
            players_data.append(player_data)
        
        #Convert to dataframe
        players_df = pd.DataFrame(players_data)
        
        players_df['Team_Name'] = team_name
        players_df['SEASON'] = year_number

        #Append to master data
        team_list_df = pd.concat([team_list_df, players_df], ignore_index=True)

#Remove NA rows
team_list_df = team_list_df.dropna(subset=['player_name'])

# Convert the DOB to datetime and extract the year
def custom_date_parser(date_str):
    # Check if date_str is missing or empty
    if not date_str or pd.isna(date_str):
        return pd.NaT  # Return Not-a-Time for missing or empty date strings
    
    day, month, year = date_str.split()
    year = int(year)
    # Adjust the century based on the year
    if year < 30:
        year += 2000
    else:
        year += 1900
    # Reformat the date string with the adjusted year
    new_date_str = f"{day} {month} {year}"
    # Convert to datetime
    return pd.to_datetime(new_date_str, format='%d %b %Y')

# Apply the custom parser to the 'DOB' column
team_list_df['date_of_birth'] = team_list_df['date_of_birth'].apply(custom_date_parser)
team_list_df['birth_year'] = pd.to_numeric(team_list_df['date_of_birth'].dt.year, errors='coerce').astype('Int64')

# Remove the unwanted character
team_list_df['player_name'] = team_list_df['player_name'].str.replace('Â', '')  
team_list_df['player_name'] = team_list_df['player_name'].str.replace(u'\xa0', u' ', regex=True)

#Create variations of player name
team_list_df[['player_first_name', 'player_last_name']] = team_list_df['player_name'].str.split(' ', n=1, expand=True)
team_list_df['player_first_name_3_letters'] = team_list_df['player_first_name'].str[:3]
team_list_df['player_first_initial'] = team_list_df['player_first_name'].str[:1]
team_list_df['player_last_name_lower_alpha_only'] = team_list_df['player_last_name'].str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)

# Now we will group by the team, year, first initial, and last name to find duplicates
# There are 64 rows of data with the players that have the same first iniial, last name, year and team. 
duplicates = team_list_df.groupby(['Team_Name', 'SEASON', 'player_first_initial', 'player_last_name']).filter(lambda x: len(x) > 1)

# Remove the duplicates from the original DataFrame
team_list_df = team_list_df.drop(duplicates.index)

#Save to csv
team_list_df.to_csv('team_list_df.csv',index=False)


#%%

### get fitzRoy data for match statistics, which has other fields that Champion Data doesnt have

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import rpy2.robjects as ro
#import re
#from urllib.request import urlopen
import pandas as pd
#import time
#import random
#from datetime import datetime

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)

start_season = 2000
finish_season = 2023

logger.info(f"Using R and Fitroy to download player data between {start_season} and {finish_season}")

r_code = f'''
    library(fitzRoy)
    df <- fetch_player_stats(season = {start_season}:{finish_season},source = 'fryzigg')
    write.csv(df,"fryzigg_df.csv",row.names = FALSE)
'''

#Source options: afltables (AFL tables), footywire (AFL Website), fryzigg (Squiggle)

ro.r(r_code)
logger.info(f"Success. Raw dataset saved")




#%%

### get injury_list from footywire, which has the current injury list for every team

#Note: This is the code for running the scrape locally. There is another code that has been uploaded to GCP to run it daily

#Set the current directory
import os
file_path = 'C:\\bcg-repos\\Player-Value\\data'
os.chdir(file_path) 

#Import packages
import logging
import re
from urllib.request import urlopen
import pandas as pd
import time
import random
from datetime import datetime

#Set number of columns to view in output
pd.set_option('display.max_columns', 100)

#Setup the logger
logger = logging.getLogger(__name__)
logger.info("Getting injury data from https://www.footywire.com/afl/footy/injury_list")

#Import any previous injury data
try:
    injury_df = pd.read_csv("injury_df.csv")
    print("Injury data found")   
except Exception:
    print("No injury data history")

#Specify the url for draft picks
url = "https://www.footywire.com/afl/footy/injury_list"

#Set the URL for the specified pick
print("Scraping injury list")

#Try to scrape the data, ensuring some buffer time between scrapes
try:
    #Random delay of between 1 and 5 seconds
    time.sleep(random.uniform(1, 5))
    response = urlopen(url)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("First scrape attempt successful")
    
#Otherwise, wait longer then try again
except Exception:
    time.sleep(random.uniform(20, 30))
    response = urlopen(url)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Second scrape attempt successful")
    
#Otherwise, wait even longer then try again
except Exception:
    time.sleep(random.uniform(30, 60))
    response = urlopen(url)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Third scrape attempt successful")

#Decode the data and split into lines
long_txt = response.read().decode()
data = long_txt.split('\n')

# Initialize variables to keep track of the current club and player information
current_club = ""
current_player_info = {}
players_info = []

# Regular expression patterns
club_pattern = r'<td height="28" align="center" colspan="3" class="tbtitle">(.*?) \(.*?\)</td>'
player_pattern = r'<td height="24" align="left" width="100">&nbsp; <a rel="nofollow" href=".*?">(.*?)</a></td>'
injury_pattern = r'<td align="center">(.*?)</td>'

# Iterate over each line in the data
for line in data:
    club_match = re.search(club_pattern, line)
    if club_match:
        current_club = club_match.group(1).strip()

    player_match = re.search(player_pattern, line)
    if player_match:
        current_player_info['Player Name'] = player_match.group(1).strip()
        current_player_info['Club Name'] = current_club
        current_player_info['Injury'] = ""
        current_player_info['Expected Return'] = ""
        
    injury_match = re.findall(injury_pattern, line)
    if len(injury_match) == 1 and 'Player Name' in current_player_info:
        if current_player_info['Injury'] == "":
            current_player_info['Injury'] = injury_match[0].strip()
        else:
            current_player_info['Expected Return'] = injury_match[0].strip()
    
    # If all fields for a player are found, add the player to the list
    if (current_player_info.get('Player Name') and 
        current_player_info.get('Injury') and 
        current_player_info.get('Expected Return')):
        players_info.append(current_player_info)
        current_player_info = {}

# Create a DataFrame
injury_df_current = pd.DataFrame(players_info)

#Add the timestamp of the data
injury_df_current['Datetime'] = timestamp

#Append to master data
try:
    injury_df = pd.concat([injury_df, injury_df_current], ignore_index=True)
    print("Appended to previous history")   
except Exception:
    injury_df = injury_df_current
    print("New file created")   

#Save to csv   
injury_df.to_csv('injury_df.csv',index=False)




