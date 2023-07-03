"""
SIS Coding Assessment
By: Jacob M. Lundeen
Date: 20230517
"""

import pandas as pd
import numpy as np
from plotnine import *
from typing import *
import random
import statsmodels.formula.api as smf

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
    before_AS = data.query('ALL_STAR_BREAK == "Before"')[['PLAYER_NAME', 'MIN', 'GAME_ID']]
    before_AS = before_AS.groupby('PLAYER_NAME').agg({'GAME_ID': 'count', 'MIN': 'sum'}).reset_index(). \
        rename(columns={'GAME_ID': 'GAMES', 'MIN': 'TOTAL_MIN'})
    before_AS['MPG'] = before_AS['TOTAL_MIN'] / before_AS['GAMES']
    before_players = before_AS.query('GAMES > 15 & MPG > 15')['PLAYER_NAME'].unique()
    after_AS = data.query('ALL_STAR_BREAK == "After"')
    after_AS = after_AS.groupby('PLAYER_NAME').agg({'GAME_ID': 'count', 'MIN': 'sum'}).reset_index(). \
        rename(columns={'GAME_ID': 'GAMES', 'MIN': 'TOTAL_MIN'})
    after_AS['MPG'] = after_AS['TOTAL_MIN'] / after_AS['GAMES']
    after_players = after_AS.query('GAMES > 15 & MPG > 15')['PLAYER_NAME'].unique()
    player_list = np.intersect1d(before_players, after_players).tolist()
    player_df = data.query('PLAYER_NAME == @player_list')
    player_df = after_as_ppg(player_df)
    return player_df


# after_as_ppg() is a helper function to all_star_players() to determine which players averaged over 15 PPG after the
# all star break.
def after_as_ppg(player_df: List[List]) -> List[List]:
    asb_df = player_df.query('ALL_STAR_BREAK == "After"')
    asb_df = asb_df.groupby('PLAYER_NAME').agg({'GAME_ID': 'count', 'PTS': 'sum'}).reset_index(). \
        rename(columns={'GAME_ID': 'GAMES', 'PTS': 'TOTAL_POINTS'})
    asb_df['PPG'] = asb_df['TOTAL_POINTS'] / asb_df['GAMES']
    asb_df['AAS_PPG'] = np.where(asb_df['PPG'] > 15, 1, 0)
    player_df = player_df.loc[:, ['FG_PCT', 'FG3A', 'FTA', 'AST', 'TOV', 'OREB', 'PLAYER_NAME']]
    player_df = player_df.groupby('PLAYER_NAME', as_index=False).mean()
    player_df['AAS_PPG'] = player_df['PLAYER_NAME'].map(dict(zip(asb_df.PLAYER_NAME, asb_df.AAS_PPG)))
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
    plt1 = ggplot(df, aes(x='PTS')) + \
           geom_histogram(bins=30) + \
           facet_wrap('TEAM', ncol=5) + \
           labs(
               y="Frequency",
               x="Points",
               title="NBA Eastern Conference Player Point Distributions",
               subtitle="Games in December 2021 Only"
           ) + \
           theme_bw() + \
           theme(figure_size=(20, 10), plot_title=element_text(size=12, face='bold'),
                 plot_subtitle=element_text(face='italic'))
    ggsave(plt1, filename='team_points_dist.png')


def main():
    df = pd.read_csv("nba_player_game_logs.csv")
    df = preprocess(df)
    avg_stats(df)
    playoff_teams(df)
    all_star(df)
    hist_plots(df)


if __name__ == "__main__":
    main()
