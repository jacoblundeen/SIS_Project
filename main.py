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
import statsmodels.formula.api as smf

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
                                'MIN', 'PLAYER_NAME', 'PLAYOFFS', 'ALL_STAR_BREAK'], inplace=True)
        player_df = player_df.sum(axis=0).div(28)
        player_df = pd.DataFrame(player_df).transpose()
        player_df['PLAYER_ID'] = player

        player_df = player_df.loc[:,
                    ['PLAYER_ID', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST',
                     'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT',
                     'FT_PCT']]
        df_28 = pd.concat([df_28, player_df], ignore_index=True)
    df_28.to_csv('players_per_28.csv', index=False)


# preprocess() identifies which games were in the regular season, the play-in tournament, the playoffs, and before/after
# the all star break.
def preprocess(data: List[List]) -> List[List]:
    reg_date = datetime.datetime(2022, 4, 12)
    playoff_date = datetime.datetime(2022, 4, 15)
    all_star_start = datetime.datetime(2022, 2, 18)
    format = '%Y-%m-%d'
    dates = data['game_date']
    playoffs = []
    all_star = []
    for date in dates:
        date = datetime.datetime.strptime(date, format)
        if date < reg_date:
            playoffs.append('No')
        elif date > playoff_date:
            playoffs.append('Yes')
        else:
            playoffs.append('None')
        if date < all_star_start:
            all_star.append('Before')
        else:
            all_star.append('After')
    data['PLAYOFFS'] = playoffs
    data['ALL_STAR_BREAK'] = all_star
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


# all_star_players() is a helper function for all_star() to create the data set for the logistic regression.
def all_star_players(data: List[List]) -> List[List]:
    players = data['PLAYER_NAME'].unique()
    player_list = []
    for player in players:
        games_before = data.query('ALL_STAR_BREAK == "Before" & PLAYER_NAME == @player')
        games_after = data.query('ALL_STAR_BREAK == "After" & PLAYER_NAME == @player')
        if games_before.shape[0] == 0:
            mpg_before = 0
        else:
            mpg_before = games_before['MIN'].sum() / games_before.shape[0]
        if games_after.shape[0] == 0:
            mpg_after = 0
        else:
            mpg_after = games_after['MIN'].sum() / games_after.shape[0]
        if games_before.shape[0] > 15 and games_after.shape[0] > 15 and mpg_before > 15 and mpg_after > 15:
            player_list.append(player)
    player_df = data.query('PLAYER_NAME == @player_list')
    player_df = after_as_ppg(player_df, player_list)
    return player_df


# after_as_ppg() is a helper function to all_star_players() to determine which players averaged over 15 PPG after the
# all star break.
def after_as_ppg(player_df: List[List], player_list: List) -> List[List]:
    ppg = {}
    for player in player_list:
        player_ppg = player_df.query('ALL_STAR_BREAK  == "After" & PLAYER_NAME == @player')
        plyr_ppg = player_ppg['PTS'].sum() / player_ppg.shape[0]
        if plyr_ppg > 15:
            ppg.update({player: 1})
        else:
            ppg.update({player: 0})
    player_df = player_df.loc[:, ['FG_PCT', 'FG3A', 'FTA', 'AST', 'TOV', 'OREB', 'PLAYER_NAME']]
    player_df = player_df.groupby('PLAYER_NAME', as_index=False).mean()
    player_df['AAS_PPG'] = player_df['PLAYER_NAME'].map(ppg)
    return player_df


# all_star() runs logistic regression on the data set and output the statistically significant predictors to a CSV file.
def all_star(data: List[List]):
    all_stars_df = all_star_players(data)
    all_stars_df = all_stars_df.sample(frac=1).reset_index(drop=True)
    model = smf.logit("AAS_PPG ~ FG_PCT + FG3A + FTA + AST + TOV + OREB", data=all_stars_df).fit()
    p_values = model.pvalues
    p_values = p_values[p_values < 0.05]
    p_values.drop(labels="Intercept", inplace=True)
    p_values = pd.DataFrame({'PREDICTOR': p_values.index, 'P_VALUE': p_values.values})
    p_values.to_csv("logistic_results.csv")


def main():
    df = pd.read_csv("nba_player_game_logs.csv")
    df = preprocess(df)
    # avg_stats(df)
    # playoff_teams(df)
    # all_star(df)


if __name__ == "__main__":
    main()
