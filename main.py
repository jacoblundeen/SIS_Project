"""
SIS Coding Assessment
By: Jacob M. Lundeen
Date: 20230517
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
import random
# import statsmodels.formula.api as smf

random.seed(49)


# avg_stats() calculates the average stats, per 28 minutes, for every player in their last 5 games of the regular
# season.
def avg_stats(data: List[List]):
    player_df = data.query('PLAYOFFS == "No"')
    player_df = player_df.sort_values('game_date').groupby('PLAYER_NAME').tail(5)
    player_df.drop(columns=['GAME_ID', 'game_date', 'TEAM', 'TEAM_ID', 'PERIOD', 'MIN', 'PLAYOFFS', 'ALL_STAR_BREAK',
                            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLAYER_NAME'], inplace=True)
    player_df = player_df.groupby('PLAYER_ID').sum().div(28)
    player_df['FG_PCT'] = np.where(player_df['FGA'] > 0, player_df['FGM'] / player_df['FGA'], 0)
    player_df['FG3_PCT'] = np.where(player_df['FG3A'] > 0, player_df['FG3M'] / player_df['FG3A'], 0)
    player_df['FT_PCT'] = np.where(player_df['FTA'] > 0, player_df['FTM'] / player_df['FTA'], 0)
    player_df.reset_index(inplace=True)
    player_df = player_df.rename(columns={'index': 'PLAYER_ID'})
    player_df.to_csv('players_per_28.csv', index=False)


# preprocess() identifies which games were in the regular season, the play-in tournament, the playoffs, and before/after
# the all star break.
def preprocess(data: List[List]) -> List[List]:
    playoff_conditions = [(data['game_date'] < '2022-04-11'),
                          (data['game_date'] > '2022-04-11') & (data['game_date'] < '2022-04-15'),
                          (data['game_date'] > '2022-04-15')]
    playoff_values = ['No', 'None', 'Yes']
    allstar_conditions = [(data['game_date'] < '2022-02-18'),
                          (data['game_date'] > '2002-02-23')]
    allstar_value = ['Before', 'After']
    data['PLAYOFFS'] = np.select(playoff_conditions, playoff_values)
    data['ALL_STAR_BREAK'] = np.select(allstar_conditions, allstar_value)
    return data


# playoff_teams() determines which players only played during the regular season and which only played during the
# playoffs and outputs those lists to CSV files.
def playoff_teams(data: List[List]):
    playoff_players = data.query('PLAYOFFS == "Yes"')['PLAYER_NAME'].unique()
    regular_players = data.query('PLAYOFFS == "No"')['PLAYER_NAME'].unique()
    regs = np.setdiff1d(regular_players, playoff_players).tolist()
    plays = np.setdiff1d(playoff_players, regular_players).tolist()
    regular_df = data.query('PLAYER_NAME == @regs')[['PLAYER_ID', 'PLAYER_NAME']]. \
        drop_duplicates(subset=['PLAYER_NAME'])
    playoff_df = data.query('PLAYER_NAME == @plays')[['PLAYER_ID', 'PLAYER_NAME']]. \
        drop_duplicates(subset=['PLAYER_NAME'])
    regular_df.to_csv('regular_season_only.csv', index=False)
    playoff_df.to_csv('playoffs_only.csv', index=False)


# all_star_players() is a helper function for all_star() to create the data set for the logistic regression.
def all_star_players(data: List[List]) -> List[List]:
    players = data['PLAYER_NAME'].unique()
    df = data.query('PLAYOFFS != None')
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


# hist_plots() creates the faceted histograms to display the points per game of the Eastern conference teams during
# December 2021.
def hist_plots(data: List[List]):
    east_conf = ['ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR',
                 'WAS']
    df = data.query('TEAM == @east_conf')
    df = df[(df['game_date'] >= '2021-12-01') & (df['game_date'] <= '2021-12-31')]
    df = df[['PTS', 'PLAYER_NAME', 'TEAM']]
    df = df.groupby(['TEAM', 'PLAYER_NAME'], as_index=False).sum()
    count = 0
    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(15, 15))
    for row in axes:
        for col in row:
            if count > 14:
                col.set_axis_off()
                break
            team = east_conf[count]
            points_df = df.query('TEAM == @team')
            col.hist(points_df['PTS'], bins=10)
            col.set_xlabel('Total Points per Player')
            col.set_ylabel(east_conf[count])
            count += 1
    fig.suptitle('Total Points per Player of Eastern Conference Teams During December 2021', fontsize=25)
    fig.tight_layout()
    plt.savefig('team_points_dist.png')


def main():
    df = pd.read_csv("nba_player_game_logs.csv")
    df = preprocess(df)
    # avg_stats(df)
    # playoff_teams(df)
    all_star(df)
    # hist_plots(df)


if __name__ == "__main__":
    main()
