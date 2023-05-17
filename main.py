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

def avg_stats(data: List[List]) -> List[List]:
    player_ids = data['PLAYER_ID'].unique()
    player_df = data[data['PLAYER_ID'] == player_ids[0]]
    player = player_df.sort_values(by='game_date', ascending=False).head(5)
    player.drop(columns=['GAME_ID', 'game_date', 'TEAM', 'TEAM_ID', 'PERIOD', 'PLAYER_ID',
                         'MIN', 'PLAYER_NAME'], inplace=True)
    player = player.sum(axis=0).div(28)
    player['PLAYER_ID'] = player_ids[0]
    player = pd.DataFrame(player).transpose()
    player = player.loc[:, ['PLAYER_ID', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST',
                            'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'FT_PCT']]
    print(player)

def playoffs(data: List[List]) -> List[List]:
    reg_date = datetime.datetime(2022, 5, 12)
    playoff_date = datetime.datetime(2022, 5, 15)
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

def main():
    df = pd.read_csv("nba_player_game_logs.csv")
    df = playoffs(df)
    avg_stats(df)

if __name__ == "__main__":

    main()