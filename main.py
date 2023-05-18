"""
SIS Coding Assessment
By: Jacob M. Lundeen
Date: 20230517
"""

import pandas as pd
import numpy as np
import math
from typing import List, Tuple, Dict, Callable
import datetime
import random

random.seed(49)

# avg_stats() calculates the average stats, per 28 minutes, for every player in their last 5 games of the regular
# season.
def avg_stats(data: List[List]):
    df_28 = pd.DataFrame(columns=['PLAYER_ID', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST',
                                  'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT',
                                  'FT_PCT'])
    player_df = data.query('PLAYOFFS == "No"')
    player_ids = player_df['PLAYER_ID'].unique()
    for player in player_ids:
        player_df = data[data['PLAYER_ID'] == player]
        player_df = player_df.sort_values(by='game_date', ascending=False).head(5)
        player_df.drop(columns=['GAME_ID', 'game_date', 'TEAM', 'TEAM_ID', 'PERIOD', 'PLAYER_ID',
                             'MIN', 'PLAYER_NAME', 'PLAYOFFS'], inplace=True)
        player_df = player_df.sum(axis=0).div(28)
        player_df = pd.DataFrame(player_df).transpose()
        player_df['PLAYER_ID'] = player

        player_df = player_df.loc[:, ['PLAYER_ID', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST',
                                'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT',
                                'FT_PCT']]
        df_28 = pd.concat([df_28, player_df], ignore_index=True)
    df_28.to_csv('players_per_28.csv', index=False)


# playoffs() identifies which games were in the regular season, the play-in tournament, or the playoffs.
def playoffs(data: List[List]) -> List[List]:
    reg_date = datetime.datetime(2022, 4, 12)
    playoff_date = datetime.datetime(2022, 4, 15)
    format = '%Y-%m-%d'
    dates = data['game_date']
    playoffs = []
    for date in dates:
        date = datetime.datetime.strptime(date, format)
        if date < reg_date:
            playoffs.append('No')
        elif date > playoff_date:
            playoffs.append('Yes')
        else:
            playoffs.append('None')
    data['PLAYOFFS'] = playoffs
    return data

# playoff_teams() determines which players only played during the regular season and which only played during the
# playoffs and outputs those lists to CSV files.
def playoff_teams(data: List[List]):
    regular_df = pd.DataFrame(columns=['PLAYER_ID', 'PLAYER_NAME'])
    playoff_df = pd.DataFrame(columns=['PLAYER_ID', 'PLAYER_NAME'])
    teams = data.query('PLAYOFFS == "Yes"')['TEAM'].unique()
    for team in teams:
        players = data[data['TEAM'] == team]
        regular_players = players.query('PLAYOFFS == "No"')
        regular_players = regular_players['PLAYER_NAME'].unique()
        playoff_players = players.query('PLAYOFFS == "Yes"')
        playoff_players = playoff_players['PLAYER_NAME'].unique()
        regs = np.setdiff1d(regular_players, playoff_players)
        plays = np.setdiff1d(playoff_players, regular_players)
        regular_df = pd.concat([regular_df, pd.DataFrame(find_player_id(data, regs))], ignore_index=True)
        playoff_df = pd.concat([playoff_df, pd.DataFrame(find_player_id(data, plays))], ignore_index=True)
    regular_df.to_csv('regular_season_only.csv', index=False)
    playoff_df.to_csv('playoffs_only.csv', index=False)

# find_player_id() is a helper function for playoff_teams() to find the unique ID for each player.
def find_player_id(data: List[List], players: List) -> Dict:
    player_list = []
    id_list = []
    if not np.any(players):
        return {'PLAYER_ID': id_list, 'PLAYER_NAME': player_list}
    else:
        for player in players:
            id = data[data['PLAYER_NAME'] == player]['PLAYER_ID'].head(1)
            player_list.append(player)
            id_list.append(id.iloc[0])
    return {'PLAYER_ID': id_list, 'PLAYER_NAME': player_list}

def main():
    df = pd.read_csv("nba_player_game_logs.csv")
    df = playoffs(df)
    # avg_stats(df)
    # playoff_teams(df)

if __name__ == "__main__":

    main()