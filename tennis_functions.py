from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from math import exp, floor
import time
from matplotlib import pyplot as plt
import requests
from lxml.html import fromstring
import os
from tqdm import tqdm
from typing import Tuple
import warnings
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error

#warnings.filterwarnings("ignore", message="The default value of regex will change from True to False in a future version.")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def player_id(names,countries,birthdays):
    player_ids = pd.Series([""] * len(names))
    bool1 = (~countries.isna()) & (~birthdays.isna()) # country and birthday both available
    bool2 = (~countries.isna()) & (birthdays.isna()) # country available, birthday not available
    bool3 = (countries.isna()) & (~birthdays.isna()) # country not available, birthday available
    bool4 = (countries.isna()) & (birthdays.isna()) # country and birthday both not available
    player_ids[bool1] = names.loc[bool1].str.lower() + "-" + countries.loc[bool1].str.slice(stop=2).str.lower() + "-" + birthdays.loc[bool1].dt.year.astype(str).str.slice(start=2)
    player_ids[bool2] = names.loc[bool2].str.lower() + "-" + countries.loc[bool2].str.slice(stop=2).str.lower()
    player_ids[bool3] = names.loc[bool3].str.lower() + "-" + birthdays.loc[bool3].dt.year.astype(str).str.slice(start=2)
    player_ids[bool4] = names.loc[bool4].str.lower()
    return player_ids
    

def tourney_id(years,names,dates):
    tourney_ids = pd.Series([""] * len(names))
    bool1 = (~dates.isna()) # date is available
    bool2 = (dates.isna()) # date is not available
    tourney_ids[bool1] = years[bool1].astype(str) + "-" + names[bool1].str.lower() + "-" + dates[bool1].dt.strftime('%Y%m%d')
    tourney_ids[bool2] = years[bool2].astype(str) + "-" + names[bool2].str.lower()
    return tourney_ids

def scores_clean(scores):
    scores2 = scores.str.replace(pat = r"\(.*\)", repl = "", regex = True) # E.g. make "7-6(3)" into "7-6"
    scores2 = scores2.str.replace(pat = "[", repl = "", regex = False) # E.g. make "[10-5]" into "10-5"
    scores2 = scores2.str.replace(pat = "]", repl = "", regex = False)
    return scores2

def match_id(tourney_ids,winner_ids,loser_ids,scores):
    return tourney_ids + "-" + winner_ids + "-" + loser_ids + scores_clean(scores)

matches_dtypes = {"winner_id": "str", "loser_id": "str", "loser_name": "str", \
    "score": "str", "best_of": "Int64", "round": "str", "minutes": "Int64", "w_ace": "Int64", "w_svpt": "Int64", "w_1stIn": "Int64", \
    "w1stWon": "Int64", "w_2ndWon": "Int64", "w_SvGms": "Int64", "w_bpFaced": "Int64", "l_ace": "Int64", "l_svpt": "Int64", "l_1stIn": "Int64", \
    "l1stWon": "Int64", "l_2ndWon": "Int64", "l_SvGms": "Int64", "l_bpFaced": "Int64", "tourney_id": "str", "match_id": "str", \
    "date": "str", "temp": "float", "wind": "float", "hum": "float"}

tournaments_dtypes = {"name": "str", "year": "Int64", "tourney_level": "str", "surface": "str", "outdoor": "str", \
    "city": "str", "venue": "str", "link": "str"}

players_dtypes = {"id": "str", "name": "str", "given_name": "str", "surname": "str", "alt_name": "str", "id_sackmann": "Int64", "country": "str", "hand": "str", \
    "height": "float", "url": "str", "url2": "str", "surname_tennis_explorer": "str", "surname_tennis_explorer2": "str", \
    "elo_overall": "float", "match_number": "Int64", "elo_hard": "float", "match_number_hard": "Int64", \
    "elo_clay": "float", "match_number_clay": "Int64", "elo_grass": "float", "match_number_grass": "Int64", "previous_match": "str"}

def read_matches() -> pd.DataFrame:
    M1 = pd.read_csv("matches_10_15.csv",dtype=matches_dtypes,parse_dates=["date"], encoding = "ISO-8859-1")
    M2 = pd.read_csv("matches_16_end.csv",dtype=matches_dtypes,parse_dates=["date"], encoding = "ISO-8859-1")
    M = pd.concat([M1,M2]).reset_index(drop=True)
    M["match_id"] = match_id(M.tourney_id,M.winner_id,M.loser_id,M.score)
    if M["match_id"].isnull().sum() > 0:
        warnings.warn("The column \match_id\" in the matches table has missing values.", UserWarning, stacklevel=2)
    M["date"] = pd.to_datetime(M["date"])
    duplicate_ids = M.loc[M.duplicated(["match_id"]),"match_id"]
    if len(duplicate_ids)>10:
        duplicate_ids.to_csv("duplicate_match_ids.csv",index=False)
        warningmessage = "The column \"id\" in the matches table is intended as a unique identifier but it has duplicate values. The duplicate values have been written to the file \"duplicate_match_ids.csv\". You should check the contents of the files and resolve the duplicates."
        warnings.warn(warningmessage, UserWarning, stacklevel=2)
    elif 1 <= len(duplicate_ids) <= 9:
        warningmessage = "The column \"match_id\" in the matches table is intended as a unique identifier but it has duplicate values. The duplicates are:"
        for duplicate in duplicate_ids:
            warningmessage += ("\n"+duplicate)
            warnings.warn(warningmessage, UserWarning, stacklevel=2)
    return M

def read_tournaments(filename: str = "tournaments.xlsx") -> pd.DataFrame:
    T = pd.read_excel(filename,dtype=tournaments_dtypes,parse_dates=["date"],sheet_name="tournaments")
    T["id"] = tourney_id(T.year,T.name,T.date)
    duplicate_ids = T.loc[T.duplicated(["id"]),"id"]
    if len(duplicate_ids)>10:
        duplicate_ids.to_csv("duplicate_tourney_ids.csv",index=True)
        warningmessage = "The column \"id\" in the table {} is intended as a unique identifier but it has duplicate values. The duplicate values have been written to the file \"duplicate_tourney_ids.csv\". You should check the contents of the files and resolve the duplicates.".format(filename)
        warnings.warn(warningmessage, UserWarning, stacklevel=2)
    elif 1 <= len(duplicate_ids) <= 9:
        warningmessage = "The column \"id\" in the table {} is intended as a unique identifier but it has duplicate values. The duplicates are:".format(filename)
        for duplicate in duplicate_ids:
            warningmessage += ("\n"+duplicate)
            warnings.warn(warningmessage, UserWarning, stacklevel=2)
    return T

def read_elos() -> pd.DataFrame:
    elos_dtypes = {"winner_elo": "float", "loser_elo": "float", "winner_elo_surface": "float", "loser_elo_surface": "float"}
    E = pd.read_csv('elos.csv',dtype=elos_dtypes)
    return E

def read_players() -> pd.DataFrame:
    P = pd.read_excel("players.xlsx", sheet_name = "players", dtype = players_dtypes, parse_dates = ["birthday"])
    duplicate_ids = P.loc[P.duplicated(["id"]),"id"]
    if len(duplicate_ids)>10:
        duplicate_ids.to_csv("duplicate_player_ids.csv",index=False)
        warningmessage = "The column \"id\" in the table players.xlsx is intended as a unique identifier but it has duplicate values. The duplicate values have been written to the file \"duplicate_player_ids.csv\". You should check the contents of the files and resolve the duplicates."
        warnings.warn(warningmessage, UserWarning, stacklevel=2)
    elif 1 <= len(duplicate_ids) <= 9:
        warningmessage = "The column \"id\" in the table players.xlsx is intended as a unique identifier but it has duplicate values. The duplicates are:"
        for duplicate in duplicate_ids:
            warningmessage += ("\n"+duplicate)
            warnings.warn(warningmessage, UserWarning, stacklevel=2)
    return P

def read_cities() -> pd.DataFrame:
    cities_dtypes = {"country": "str", "city": "str", "lat": "float", "long": "float", "elev": "float", "time_difference": "str", \
        "time_difference_direction": "str", "close_city": "str", "URL": "str", "URL2": "str", "URL3": "str", "relevant_for_weather": "bool"}
    C = pd.read_csv("cities.csv",dtype=cities_dtypes)
    if C["city"].duplicated().any():
        warnings.warn("The column \"city\" in the table cities.csv is intended as a unique identifier but it has duplicate values.", UserWarning, stacklevel=2)
    return C 

def read_college_matches() -> pd.DataFrame:
    return pd.read_excel("college_tennis.xlsx",sheet_name="matches",dtype=matches_dtypes,parse_dates=["date"])

def read_scores_matches(filename: str) -> pd.DataFrame:
    return pd.read_excel(filename + ".xlsx",sheet_name="tournaments",dtype=tournaments_dtypes)

def read_scores_tournaments(filename: str) -> pd.DataFrame:
    return pd.read_excel(filename + ".xlsx",sheet_name="tournaments",dtype=tournaments_dtypes)

def winprobabilities_from_elo(surfaces: pd.Series, best_ofs: pd.Series, w_elo_before: pd.Series, l_elo_before: pd.Series, w_elo_before_surf: pd.Series, \
                              l_elo_before_surf: pd.Series, tourney_levels: pd.Series, w_hard: float = 0.281, w_clay: float = 0.362, w_grass_low: float = 0.206, w_grass_high: float = 0.266) -> pd.DataFrame:
    """
    This function first calculates the win probabilities according to the Elos from before the match. A combination of the probability according to overall Elo and surface specific Elo is used. To increase performance, a different weight is used for each surface (e.g. there are very few matches played on grass courts, so the weight on surface Elo is lower here due to lower sample size).

    Args:
        best_ofs (pandas series): 3 for best-of-3 matches and 5 for best-of-5 matches
        w_elo_before (pandas series): The overall Elos of the winner before the start of the match
        l_elo_before (pandas series): The overall Elos of the loser before the start of the match
        w_elo_before_surf (pandas series): The surface Elos of the winner before the start of the match
        l_elo_before_surf (pandas series): The surface Elos of the loser before the start of the match
        tourney_levels (pandas series): The tourney_levels of the tournament the match belongs to
        w_hard (float): weight assigned to the surface specific win probability for hard court matches
        w_clay (float): same as w_hard but for clay court matches
        w_grass_low (float): same as w_hard but for grass court matches below tour level
        w_grass_high (float): same as w_hard but for grass court matches of tour level
    
    Returns:
        w_winprob_comb (pandas series): The estimated probability that the winner wins the match with only the information from w_elo_before, l_elo_before, w_elo_before_surf and l_elo_before_surf
    """

    M = pd.DataFrame({"surface": surfaces, "best_of": best_ofs, "tourney_level": tourney_levels, "winner_elo_before": w_elo_before, "loser_elo_before": l_elo_before, \
                      "winner_elo_surface_before": w_elo_before_surf, "loser_elo_surface_before": l_elo_before_surf}) # for convenience, put everything in dataframe
    
    # Step 1 is to calculate the setwinprob. via standard formula cf. https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details.
    M["winner_setwinprob_overall"] = 1/(1+10**( (M["loser_elo_before"]-M["winner_elo_before"]) /400))
    M["winner_setwinprob_surface"] = 1/(1+10**( (M["loser_elo_surface_before"]-M["winner_elo_surface_before"]) /400))
    
    # Step 2 is to convert to matchwinprob. Obviously, it depends on best_of == 5 or best_of == 3. The formula can be derived via tree diagram.
    best_of_3 = (M["best_of"] == 3)
    best_of_5 = (M["best_of"] == 5)

    M.loc[best_of_3,"winner_winprob_overall"] = M.loc[ best_of_3, "winner_setwinprob_overall"]**2+\
        2*M.loc[ best_of_3, "winner_setwinprob_overall"]**2*(1-M.loc[ best_of_3, "winner_setwinprob_overall"])
    
    M.loc[best_of_5,"winner_winprob_overall"] = M.loc[ best_of_5, "winner_setwinprob_overall"]**3+ \
        3*M.loc[ best_of_5, "winner_setwinprob_overall"]**3*(1-M.loc[ best_of_5, "winner_setwinprob_overall"])+ \
        6*M.loc[ best_of_5, "winner_setwinprob_overall"]**3*(1-M.loc[ best_of_5, "winner_setwinprob_overall"])**2
    
    M.loc[best_of_3,"winner_winprob_surface"] = M.loc[best_of_3, "winner_setwinprob_surface"]**2+\
        2*M.loc[best_of_3, "winner_setwinprob_surface"]**2*(1-M.loc[best_of_3, "winner_setwinprob_surface"])
    
    M.loc[best_of_5,"winner_winprob_surface"] = M.loc[best_of_5, "winner_setwinprob_surface"]**3+ \
        3*M.loc[best_of_5, "winner_setwinprob_surface"]**3*(1-M.loc[best_of_5, "winner_setwinprob_surface"])+ \
        6*M.loc[best_of_5, "winner_setwinprob_surface"]**3*(1-M.loc[best_of_5, "winner_setwinprob_surface"])**2
    
    # Step 3 is to combine overall and surface specific winprobabilities
    M["w_surface"] = 0 # Initialize as 0
    M.loc[ M["surface"] == "Hard", "w_surface" ] = w_hard
    M.loc[ M["surface"] == "Clay", "w_surface" ] = w_clay
    high_tourney_levels={"ATP 1000","ATP 250","ATP 500","Minals","Grand Slam","Olympics"}
    M.loc[ ((M["surface"]=="Grass") | (M["surface"]=="Carpet") & (M["tourney_level"].isin(high_tourney_levels))), "w_surface" ] = w_grass_high
    M.loc[ ((M["surface"]=="Grass") | (M["surface"]=="Carpet") & ~(M["tourney_level"].isin(high_tourney_levels))), "w_surface" ] = w_grass_low

    M["winner_winprob_comb"] = M["w_surface"] * M["winner_winprob_surface"] + (1-M["w_surface"]) * M["winner_winprob_overall"]

    return M["winner_winprob_comb"]

def tourney_id_to_year(tourney_ids: pd.Series):
    return (tourney_ids.str.slice(start=0,stop=4)).astype(int)
    

def create_master_table(previous_columns_bool: bool = False, cities_bool: bool = False, matches_table: pd.DataFrame = read_matches(), tournaments_table: pd.DataFrame = read_tournaments(), cities_table: pd.DataFrame = read_cities()) -> pd.DataFrame:
    """
    
    Args:
        M (pandas dataframe): corresponds to matches.csv
        T (pandas dataframe): corresponds to tournaments.csv
        E (pandas dataframe): corresponds to elos.csv
        C (pandas dataframe): corresponds to cities.csv
        previous_columns_bool (boolean): If true, then the columns "winner_previous_surface", "loser_previous_surface", "winner_previous_won", "loser_previous_won", "winner_previous_score", "loser_previous_score" will be included in the final table
    
    Returns:
        the table M joined with T, E and C
    """

    tournaments_table = tournaments_table.rename({"id": "tourney_id", "name": "tourney_name", "date": "tourney_date", "level": "tourney_level"}, axis=1)
        
    master_table = pd.merge(matches_table, tournaments_table, how = "left", left_on = "tourney_id", right_on = "tourney_id")

    if cities_bool:
        C = cities_table.loc[:,["city","country","lat","long","elev"]]
        master_table = pd.merge(master_table, C, how = "left", left_on = "city", right_on = "city")

    master_table["year"] = tourney_id_to_year(master_table["tourney_id"])

    if previous_columns_bool:
        matches_lookup = master_table.loc[:,["match_id","surface","winner_id","score","winner_elo","winner_elo_surface","loser_elo","loser_elo_surface"]]
        matches_lookup_winner = matches_lookup.rename({"match_id": "winner_previous_match", "surface": "winner_previous_surface", "winner_id": "winner_previous_winner", \
            "score": "winner_previous_score", "winner_elo": "winner_previous_winner_elo", "winner_elo_surface": "winner_previous_winner_elo_surface", \
            "loser_elo": "winner_previous_loser_elo", "loser_elo_surface": "winner_previous_loser_elo_surface"}, axis=1)
        master_table = pd.merge(master_table,matches_lookup_winner,how="left",on="winner_previous_match")

        matches_lookup_loser = matches_lookup.rename({"match_id": "loser_previous_match", "surface": "loser_previous_surface", "winner_id": "loser_previous_winner", \
           "score": "loser_previous_score", "winner_elo": "loser_previous_winner_elo", "winner_elo_surface": "loser_previous_winner_elo_surface", \
            "loser_elo": "loser_previous_loser_elo", "loser_elo_surface": "loser_previous_loser_elo_surface"}, axis=1)
        master_table = pd.merge(master_table,matches_lookup_loser,how="left",on="loser_previous_match")

        master_table["winner_previous_won"] = (master_table["winner_id"]==master_table["winner_previous_winner"])
        master_table.loc[master_table["winner_previous_match"].isnull(),"winner_previous_won"] = np.nan

        master_table["loser_previous_won"] = (master_table["loser_id"]==master_table["loser_previous_winner"])
        master_table.loc[master_table["loser_previous_match"].isnull(),"loser_previous_won"] = np.nan

        master_table["winner_elo_before"] = np.where(master_table['winner_previous_won'], master_table["winner_previous_winner_elo"], np.where(master_table['winner_previous_won'] == False, master_table["winner_previous_loser_elo"], np.nan))
        master_table["loser_elo_before"] = np.where(master_table['loser_previous_won'], master_table["loser_previous_winner_elo"], np.where(master_table['loser_previous_won'] == False, master_table["loser_previous_loser_elo"], np.nan))
        master_table["winner_elo_surface_before"] = np.where(master_table['winner_previous_won'], master_table["winner_previous_winner_elo_surface"], np.where(master_table['winner_previous_won'] == False, master_table["winner_previous_loser_elo_surface"], np.nan))
        master_table["loser_elo_surface_before"] = np.where(master_table['loser_previous_won'], master_table["loser_previous_winner_elo_surface"], np.where(master_table['loser_previous_won'] == False, master_table["loser_previous_loser_elo_surface"], np.nan))

        master_table = master_table.drop(["winner_previous_winner", "loser_previous_winner", "winner_previous_winner_elo", \
            "winner_previous_loser_elo", "winner_previous_winner_elo_surface", "winner_previous_loser_elo_surface", \
            "loser_previous_winner_elo", "loser_previous_loser_elo", "loser_previous_winner_elo_surface", "loser_previous_loser_elo_surface"], axis=1)
        
        master_table["winner_winprob_elo"] = winprobabilities_from_elo(master_table["surface"],master_table["best_of"], master_table["winner_elo_before"], master_table["loser_elo_before"], \
            master_table["winner_elo_surface_before"], master_table["loser_elo_surface_before"], master_table["tourney_level"])

    return master_table

def prediction_metrics(winner_winprobabilities: pd.Series, years: pd.Series):
    M = pd.DataFrame({"winner_winprob_elo": winner_winprobabilities[winner_winprobabilities.notna()], "year": years[winner_winprobabilities.notna()]})
    outcomes = np.random.choice([0,1], size=len(M), p=[0.5, 0.5])
    predictions = np.where(outcomes==1, M["winner_winprob_elo"], 1-M["winner_winprob_elo"])
    brierscore = brier_score_loss(outcomes,predictions)
    log_score = log_loss(outcomes,predictions)
    mae = mean_absolute_error(outcomes,predictions)
    return brierscore, log_score, mae

def brierscore(winner_winprobabilities_comb: pd.Series, years: pd.Series):
    M = pd.DataFrame({"winner_winprob_comb": winner_winprobabilities_comb, "year": years})
    M = M.loc[~M["year"].isna(),:]
    brierscore_total = np.mean(((1-winner_winprobabilities_comb)**2))
    years_list = list(set(M["year"]))
    B = pd.DataFrame({"year": years_list, "brierscore": [0]*len(years_list)})
    
    for i in len(B.index):
        year = B.at[i,"year"]
        B.at[i,"brierscore"] = np.mean(((1-M.loc[M["year"] == year,"winner_winprob_comb"])**2))

    brierscore_alt = brier_score_loss(1,M["winner_winprob_comb"])
    log_score = log_loss(1, M["winner_winprob_comb"])
    mae = mean_absolute_error(1, M["winner_winprob_comb"])

    return brierscore_total, brierscore_alt, log_score, mae, B




def sort_matches_table(M: pd.DataFrame, startdate: datetime = datetime.strptime("2009-12-01", '%Y-%m-%d'), drop_date2: bool = True) -> pd.DataFrame:
    """This function sorts the table M in chronological order
    
    Args:
        M (pandas dataframe): corresponds to matches.csv
        startdate(datetime): the cutoff time, data older than startdate is irrelevant, default is Dec 1, 2009
        drop_date2(boolean): the variable date2 is equal to date where data is available and tourney_date else
    
    Returns:
        the data from the table M after startdate in chronological order (ascending) 
    """

    
    M.date=pd.to_datetime(M.date)
    M.tourney_date=pd.to_datetime(M.tourney_date)
    # Create auxiliary variable date2 which is equal to date if date exists and tourney_date else
    M["date2"]=M["date"]
    M.loc[pd.isna(M.date),"date2"]=M.loc[pd.isna(M.date),"tourney_date"]
    M=M.loc[pd.notna(M.date2),:]
    M=M.loc[M.date2>=startdate]
    # Create auxiliary variable roundvalue (higher value means later stage of tournament)
    rounds=["NA","Q1","Q2","Q3","Q4","R128","R64","R32","R16","QF","SF","BR","F"\
        "RR 1","RR 2","RR 3","RR"]
    d = {'round': rounds, 'roundvalue': [*range(len(rounds))]}
    df = pd.DataFrame(data=d)
    M_merged = pd.merge(M,df,how="left",on="round")
    # Sort the table by date2 and use roundvalue as tiebreaker
    M_final = M_merged.sort_values(by=["date2","roundvalue"])
    M_final = M_final.reset_index(drop=True)
    # Delete the auxiliary variables
    M_final = M_final.drop(["roundvalue"], axis=1)
    if drop_date2:
        M_final = M_final.drop(["date2"], axis=1)
    return M_final

def elos_available_years():
    """ Returns a string with all the years for which a year-end Elo rating file exists in the folder
    Note that these files need to be saved under under the name format elo_ratings_yearend_year.csv
    """

    files = os.listdir()
    i = 0
    for year in range(2009,2022):
        filename = "elo_ratings_yearend_" + str(year) + ".csv"
        if filename in files and i==0:
            available_years = str(year+1)
            i = 1
        elif filename in files and i==1:
            available_years = available_years + ", " + str(year+1)  
    return available_years

def setwinner(sets: pd.Series) -> pd.Series:
    """ Returns the winner of a completed set (either "winner" for the winner of the whole match, "loser" for the loser of the whole match or "neither" if the result of the set is not valid)
    E.g. the input "6-3" will return "winner", the input "6-7" will return "loser" and the input "3-0 RET" will return "neither"
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sets_clean = sets.str.replace(r"\(.*\)","", regex = True) # E.g. make "7-6(3)" into "7-6"
        sets_clean = sets_clean.str.replace("[","", regex = False) # E.g. make "[10-5]" into "10-5"
        sets_clean = sets_clean.str.replace("]","", regex = False)
        
    sets_df = sets_clean.str.split('-', expand = True)
    sets_df = sets_df.rename({0: "winner_games_won", 1: "loser_games_won"}, axis = 1)
    sets_df["winner_games_won"] = pd.to_numeric(sets_df.winner_games_won, errors = 'coerce').fillna(0).astype('int')
    sets_df["loser_games_won"] = pd.to_numeric(sets_df.loser_games_won, errors ='coerce').fillna(0).astype('int')

    # We define that a player wins a set if one of the following is true:
    # 1) In the final score they have won at least 6 games and they have won by a margin of at least 2
    # 2) They win 7-6 (i.e. by tiebreak)
    # Therefore we make a case distinction:
    sets_df["winner_won"] = ((sets_df.winner_games_won.astype(int) > (sets_df.loser_games_won.astype(int)+1)) & (sets_df.winner_games_won.astype(int) >= 6)) | \
        ( (sets_df.winner_games_won.astype(int) == 7) & (sets_df.loser_games_won.astype(int) == 6) )
    sets_df["loser_won"] = ((sets_df.loser_games_won.astype(int) > (sets_df.winner_games_won.astype(int)+1)) & (sets_df.loser_games_won.astype(int) >= 6)) | \
        ( (sets_df.loser_games_won.astype(int) == 7) & (sets_df.winner_games_won.astype(int) == 6) )
    
    setwinner_name = pd.Series(["neither"]).repeat(len(sets_clean)) # Initialize with "neither" everywhere
    setwinner_name.iloc[sets_df.winner_won] = "winner"
    setwinner_name.iloc[sets_df.loser_won] = "loser"

    return setwinner_name



def sets_won_by_player(scores: pd.Series):
    """Calculates the number of (complete) sets won by each player from the variable score
    E.g. the input "6-0 0-6 6-0 6-0" will return 3 and 1, the input "6-0 3-0 RET" will return 1 and 0
    
    Args:
        score (pandas series)
    
    Returns:
        the two
    """

    #scores_df = scores.str.split().apply(pd.Series) # produces a table, the entry in cell (i,j) is the i-th set of the j-th score
    scores_df = scores.str.split(' ', expand=True)
    m = len(scores_df.columns)
    winner_no_setswon = np.zeros(len(scores)).astype(int) # initialize with zeros
    loser_no_setswon = np.zeros(len(scores)).astype(int)
    # For each column (i.e. for each set) find out who won that set, using the setwinner function and add the result to winner_setswon and loser_setswon
    set_number = 1
    for set in scores_df.columns:
        if set_number > 5:
            break
        sets = scores_df.loc[:,set]
        setwinner_names = setwinner(sets)
        winner_no_setswon[setwinner_names == "winner"] = winner_no_setswon[setwinner_names == "winner"] + 1
        loser_no_setswon[setwinner_names == "loser"] = loser_no_setswon[setwinner_names == "loser"] + 1
        set_number += 1

    return winner_no_setswon, loser_no_setswon

def elo_factors(elo: float, match_number: int, recentmatches: int, penaltyfactor: float = 0.98):
    
    """
    Calculate the activityfactors (if players have few recent matches, the K factor will be higher)
    We use the formula activityfactor = exp(-0.4*recentmatches)+1
    So 0 recentmatches -> activityfactor=2, 1 recentmatches -> activityfactor=1.67, ..., 10 recentmatches -> factor=1.02, ...
    If recentmatches=0 there will also be a penalty (because players tend to be weaker after an injury break)
    """

    if match_number < 40:
        return 1,1
    else:
        activityfactor = exp(-0.4*recentmatches)+1
    if recentmatches == 0:
        penaltyfactor_indiv = 1-(1-penaltyfactor)/(1+exp(-0.05*(elo-1910))*((1/0.995)-1))
        return activityfactor, penaltyfactor_indiv
    else:
        return activityfactor, 1

def K_factor(match_number: int, c: float = 250, o: float = 20, s: float = 0.6) -> float:

    """
    The K factor determines how quickly a player's Elo rating adapts to good or bad performance. Too high and the rating will be too sensitive, too low and the rating of improving young players won't go up fast enough.
    Instead of using the same K for every player, we make it dependent on the number of matches they have played in their career.
    We use the formula from https://web.archive.org/web/20190321232517/https://www.betfair.com.au/hub/tennis-elo-modelling/.
    The formula depends on a number of parameters. We tested a large number of combinations of parameters. The default values performed well in our tests. For surface-specific elos the parameters are chosen slightly differently to increase performance.
    Args:
        match_number (integer): The total number of matches the player has played in their career
        c (float): A multiplicatory constant
        o (float): Small offset
        s (float): Shape parameter
    Returns:
        K (float): The player's K factor
    """
    return c/(match_number+o)**s

def elo_new(w_elo_old: float, l_elo_old: float, w_K: float, l_K: float, w_setswon: int, l_setswon: int):
    """
    # We use a weighted Elo algorithm (weight = number of sets the player won)
    """
    w_setwinprob = 1/ (1 + 10**((l_elo_old - w_elo_old)/400))
    w_elogain_by_setwon = w_K*(1-w_setwinprob)
    w_eloloss_by_setwon = w_K*w_setwinprob
    l_elogain_by_setwon = l_K*w_setwinprob
    l_eloloss_by_setwon = l_K*(1-w_setwinprob)

    w_elo_new = w_elo_old + w_elogain_by_setwon * w_setswon - w_eloloss_by_setwon * l_setswon
    l_elo_new = l_elo_old + l_elogain_by_setwon * l_setswon - l_eloloss_by_setwon * w_setswon
    
    return w_elo_new, l_elo_new

def winner_and_loser_row(winner_ids, loser_ids, ids_lookup):
    df_winner = pd.DataFrame({"winner_id": ids_lookup, "winner_row": range(len(ids_lookup))})
    df_loser = pd.DataFrame({"loser_id": ids_lookup, "loser_row": range(len(ids_lookup))})
    df_left = pd.DataFrame({"winner_id": winner_ids, "loser_id": loser_ids})
    df_merged = pd.merge(df_left, df_winner , how = "left", on = "winner_id")
    df_merged = pd.merge(df_merged, df_loser, how = "left", on = "loser_id")
    return df_merged.winner_row, df_merged.loser_row

def fill_player_table(ids_total: set, P: pd.DataFrame, initial_elo: float):
    """ Adds rows and columns to the dataframe P, if necessary
    """
    
    # Step 1: If surface specific columns are missing, initialize them in the following way.
    if "elo_hard" not in P.columns:
        P["elo_hard"] = P.elo_overall
        P["match_number_hard"] = 0
    if "elo_clay" not in P.columns:
        P["elo_clay"] = P.elo_overall
        P["match_number_clay"] = 0
    if "elo_grass" not in P.columns:
        P["elo_grass"] = P.elo_overall
        P["match_number_grass"] = 0
    if "previous_match" not in P.columns:
        P["previous_match"] = ""

    P.loc[P["previous_match"].isna(),"previous_match"] = ""
    
    # Step 2: Check if there are players that are not already part of the dataframe P
    ids_avail = set(P.id)
    ids_unavail = list(ids_total.difference(ids_avail))
    P_unavail = pd.DataFrame({"id": ids_unavail, "elo_overall": initial_elo, "match_number": 0, "elo_hard": initial_elo, \
        "match_number_hard": 0, "elo_clay": initial_elo, "match_number_clay": 0, "elo_grass": initial_elo, "match_number_grass": 0, \
        "previous_match": ""})
    P_all = pd.concat([P,P_unavail]).reset_index(drop=True)

    return P_all


def update_elo(match_ids: pd.Series, winner_ids: pd.Series, loser_ids: pd.Series, scores: pd.Series, tourney_dates: pd.Series, \
    dates: pd.Series, surfaces: pd.Series, tourney_levels: pd.Series, rounds: pd.Series, player_table: pd.DataFrame, c: float = 250, \
    c_hard: float = 280, c_clay: float = 300, c_grass: float = 350, o: float = 20, s1: float = 0.6, initial_elo: float = 1400, \
    recentdays_range: int = 75, penaltyfactor: float = 0.98) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """A function that calculates the Elo ratings for a given dataframe M and given parameters.

    The general principles of the Elo algorithm are explained on many sites, e.g. https://en.wikipedia.org/wiki/Elo_rating_system.
    
    Note that we use a set-based algorithm instead of a match-based. That is done to (1) increase the sample size and (2) factor in that winning by 3-0 sets is better than winning by 2-1. Therefore the Elos tell us the probability of winning a set, not the whole match.
    
    Args:
        match_ids (pd.Series): unique identifier of the matches
        winner_ids (pd.Series): player ids of the winners
        loser_ids (pd.Series): player ids of the losers
        scores (pd.Series): the results of the matches
        tourney_dates (pd.Series): the dates at which the tournaments the matches belong to started
        dates (pd.Series): the dates of the actual matches
        surfaces (pd.Series): the surfaces on which the matches were played
        tourney_levels (pd.Series): the tournament categories (e.g. Grand Slam, ATP 500 etc.)
        rounds (pd.Series): the stage of the tournament (e.g. R32, SF, F etc.)
        player_table (pd.DataFrame): a table containing a list of players and their elos at the start
        c (float): A multiplicatory constant that determines a player's K factor in the Elo algorithm
        c_hard (float): Same as c but for the hard court Elos
        c_clay (float): Same as c but for the clay court Elos
        c_grass (float): Same as c but for the grass court Elos
        o (float): Small offset, used to determine a player's K factor
        s1 (float): Shape parameter, used to determine a player's K factor
        initial_elo (float): Elo rating assigned to new players (i.e. players that do not appear in player_table)
        recentdays_range (int): If player hasn't played a single match during this number of days, then we make his rating lower
        penaltyfactor (float): Parameter to determine how severely we punish absence
    
    Returns:
        N (pandas dataframe): A match by match table with the columns "match_id", "winner_previous_match", "loser_previous_match", "winner_elo", "loser_elo", "winner_elo_surface" and "loser_elo_surface"
        P (pandas dataframe): An updated version of player_table with the final ratings and match numbers
    """

    # Create a dataframe from all the pandas series
    M = pd.DataFrame({"match_id": match_ids, "winner_id": winner_ids, "loser_id": loser_ids, "score": scores, "tourney_date": tourney_dates, \
        "date": dates, "surface": surfaces, "tourney_level": tourney_levels, "round": rounds})
    
    # Delete rows of M that do not need basic sanity checks
    M = M.loc[(M["surface"] == "Hard") | (M["surface"] == "Clay") | (M["surface"] == "Grass") | (M["surface"] =="Carpet") ,:].reset_index(drop=True) # surface missing or does not make sense
    M = M[~(M["match_id"].duplicated(keep=False))] # match_id duplicated

    # Sort the table chronologically and delete the columns "round", "date" and "tourney_date" because not needed anymore
    M = sort_matches_table(M,drop_date2=False)
    
    # Calculate sets won by each player (we need this because we use a set-based algorithm)
    M["winner_setswon"], M["loser_setswon"] = sets_won_by_player(M["score"])
    
    # Create dataframe P that contains both the players from player_table and all the other players
    ids_total = set(M["winner_id"]).union(set(M["loser_id"]))
    P = fill_player_table(ids_total,player_table,initial_elo)

    # Convert P to a numpy array (will be much faster than pandas dataframe, we will convert it back in the end)
    P_arr = np.array(P.values)

    # Create a dataframe that stores for each player the row indexes of M where that player appears
    # It will become clear later why we need this
    players_row_indexes_in_M = pd.DataFrame({"list_of_indexes": [[] for _ in range(len(P.index))]}) # Initialize with only empty lists

    # Add columns "winner_row" and "loser_row" to M (they tell us at which position in P the winner and loser are located respectively)
    M["winner_row"], M["loser_row"] = winner_and_loser_row(M["winner_id"],M["loser_id"],P["id"])

    # Add the column c_surface to the dataframe M
    M["c_surface"] = c_hard
    M.loc[M["surface"]=="Clay","c_surface"] = c_clay
    M.loc[(M["surface"]=="Grass") | (M["surface"]=="Carpet"), "c_surface"] = c_grass

    # Delete columns from M that were only needed temporarily
    M = M.drop(["score","winner_id","loser_id","round","date","tourney_date"], axis=1)

    # Initialize the dataframe N (as with P, we work with a numpy array and will convert to pandas dataframe in the end)
    N_arr = np.zeros((len(M.index), 7), dtype=object)
    N_arr[:, 0] = "tmp"
    N_arr[:,5] = "tmp"
    N_arr[:,6] = "tmp"

    # Create numpy array with the dates (we also do this because it's faster than accessing the column in M)
    dates_arr = np.array(M["date2"].values)

    # Start iterating over rows of M
    for i, row in tqdm(M.iterrows(), total=M.shape[0]):
        
        # Retrieve all the relevant data from the i-th row of M
        match_id_current, surface, tourney_level, date2, winner_setswon, loser_setswon, winner_row, loser_row, c_surface = row

        # Retrieve data from P (we know the row indexes winner_row and loser_row because we included them as columns in the table M)
        w_elo_old = P_arr[winner_row,1]
        l_elo_old = P_arr[loser_row,1]
        w_match_number = P_arr[winner_row,2]
        l_match_number = P_arr[loser_row,2]
        w_previous_match = P_arr[winner_row,9]
        l_previous_match = P_arr[loser_row,9]

        surface_elo_column_index = 3
        if surface == "Clay":
            surface_elo_column_index = 5
        elif surface == "Grass" or surface == "Carpet":
            surface_elo_column_index = 7

        w_elo_old_surface = P_arr[winner_row,surface_elo_column_index]
        l_elo_old_surface = P_arr[loser_row,surface_elo_column_index]
        w_match_number_surface = P_arr[winner_row,(surface_elo_column_index+1)]
        l_match_number_surface = P_arr[loser_row,(surface_elo_column_index+1)]
    
        # Calculate the K factors for winner and loser, both for the overall elo and the surface specific elo (so 4 different K's in total)
        w_K = K_factor(w_match_number,c,o,s1)
        l_K = K_factor(l_match_number,c,o,s1)
        w_K_surface = K_factor(w_match_number_surface,c_surface,o,s1)
        l_K_surface = K_factor(l_match_number_surface,c_surface,o,s1)
        
        # Calculate number of recent matches
        # Since going through the whole dataframe to find out the number of matches would be highly inefficient, we make use of the dataframe df:
        # In row winner_row of df, there's a list of all the row indexes where winner_id appears. So we only need to go through these indexes.
        # That way, instead of going through a dataframe of more than 300k rows in each step, we go through dataframes of at most 1k rows.
        
        date_cutoff = date2-timedelta(days=recentdays_range) # The date that was recentdays_range number of days in the past

        if w_match_number >= 40 or l_match_number >= 40: # only need to do this if at least one of the players has match_number >= 40
            if w_match_number>= 40 and l_match_number >=40:
                df_sub_winner = dates_arr[players_row_indexes_in_M.at[winner_row,"list_of_indexes"]]
                w_recentmatches = sum(df_sub_winner>=date_cutoff)
                df_sub_loser = dates_arr[players_row_indexes_in_M.at[loser_row,"list_of_indexes"]]
                l_recentmatches = sum(df_sub_loser>=date_cutoff)
            elif w_match_number >= 40 and l_match_number < 40:
                df_sub_winner = dates_arr[players_row_indexes_in_M.at[winner_row,"list_of_indexes"]]
                w_recentmatches = sum(df_sub_winner>=date_cutoff)
                l_recentmatches = 1
            elif w_match_number < 40 and l_match_number >= 40:
                w_recentmatches = 1
                df_sub_loser = dates_arr[players_row_indexes_in_M.at[loser_row,"list_of_indexes"]]
                l_recentmatches = sum(df_sub_loser>=date_cutoff)
            w_activityfactor, w_penaltyfactor = elo_factors(w_elo_old,w_match_number,w_recentmatches,penaltyfactor)
            l_activityfactor, l_penaltyfactor = elo_factors(l_elo_old,l_match_number,l_recentmatches,penaltyfactor)
            w_K *= w_activityfactor
            l_K *= l_activityfactor
            w_K_surface *= w_activityfactor 
            l_K_surface *= l_activityfactor
            w_elo_old *= w_penaltyfactor
            l_elo_old *= l_penaltyfactor
        
        # Update df by appending the row of the current match to the two lists
        players_row_indexes_in_M.at[winner_row,"list_of_indexes"].append(i)
        players_row_indexes_in_M.at[loser_row,"list_of_indexes"].append(i)
        
        # Results at exhibition tournaments are less indicative of player strength than any other tournament category.
        # Therefore, we reduce all the K factors (namely divide by 2)
        if tourney_level == "Exhibition":
            w_K //= 2
            l_K //= 2
            w_K_surface //= 2
            l_K_surface //= 2

        # Calculate the new elo ratings
        winner_elo_new, loser_elo_new = elo_new(w_elo_old, l_elo_old, w_K, l_K, winner_setswon, loser_setswon)
        winner_elo_new_surface, loser_elo_new_surface = elo_new(w_elo_old_surface, l_elo_old_surface, w_K_surface, l_K_surface, winner_setswon, loser_setswon )
        
        # Write everything to the i-th row of N
        N_arr[i] = (match_id_current, winner_elo_new, loser_elo_new, winner_elo_new_surface, loser_elo_new_surface, w_previous_match, l_previous_match)
        
        # Write everything to P
        P_arr[winner_row,1] = winner_elo_new
        P_arr[winner_row,2] = w_match_number+1
        P_arr[loser_row,1] = loser_elo_new
        P_arr[loser_row,2] = l_match_number+1
        P_arr[winner_row,9] = match_id_current
        P_arr[loser_row,9] = match_id_current
         
        P_arr[winner_row,surface_elo_column_index] = winner_elo_new_surface
        P_arr[winner_row,(surface_elo_column_index+1)] = w_match_number_surface + 1
        P_arr[loser_row,surface_elo_column_index] = loser_elo_new_surface
        P_arr[loser_row,(surface_elo_column_index+1)] = l_match_number_surface + 1

    
    N = pd.DataFrame(N_arr)
    N = N.rename({0: "match_id", 1: "winner_elo", 2: "loser_elo", 3: "winner_elo_surface", 4: "loser_elo_surface", 5: "winner_previous_match", \
        6: "loser_previous_match"}, axis=1)
    
    P = pd.DataFrame(P_arr)
    P = P.rename({0: "id", 1: "elo_overall", 2: "match_number", 3: "elo_hard", 4: "match_number_hard", 5: "elo_clay", 6: "match_number_clay", \
        7: "elo_grass", 8: "match_number_grass", 9: "previous_match"}, axis=1)
    P = P.sort_values(by=["elo_overall"], ascending = False)
    P = P.reset_index(drop=True)

    return N, P


def write_tournaments_table(T: pd.DataFrame):
    writer = pd.ExcelWriter("tournaments.xlsx", engine="xlsxwriter",datetime_format="YYYY-MM-DD")

    T.to_excel(writer, sheet_name="tournaments", startrow=1, header=False, index=False)

    workbook = writer.book
    worksheet = writer.sheets["tournaments"]

    cell_format = workbook.add_format({"font_size": 10, 'font_name': 'Arial'})
    (max_row, max_col) = T.shape

    column_settings = [{'header': column} for column in T.columns]
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings, "style": "Table Style Medium 7"})
    worksheet.set_column(0, 0, 50,cell_format)
    worksheet.set_column(1, 1, 8,cell_format) 
    worksheet.set_column(2,2,14,cell_format)
    worksheet.set_column(3,3,22,cell_format)
    worksheet.set_column(4,5,10,cell_format)
    worksheet.set_column(6,7,30,cell_format)
    worksheet.set_column(8,8,50,cell_format)
    writer.close()


def write_player_table(P: pd.DataFrame, initial_elo: float = 1400):
    
    """A function that writes the dataframe in P to the Excel file "players.xlsx".
    
    Args:
        P (pandas dataframe): Table of player information
    """
    
    P_pretty = P # pretty version of P for Excel
    P_pretty["elo_overall"] = P_pretty["elo_overall"].fillna(initial_elo).round(2)
    P_pretty["elo_hard"] = P_pretty["elo_hard"].fillna(initial_elo).round(2)
    P_pretty["elo_clay"] = P_pretty["elo_clay"].fillna(initial_elo).round(2)
    P_pretty["elo_grass"] = P_pretty["elo_grass"].fillna(initial_elo).round(2)
    P_pretty["match_number"] = P_pretty["match_number"].fillna(0)
    P_pretty["match_number_hard"] = P_pretty["match_number_hard"].fillna(0)
    P_pretty["match_number_clay"] = P_pretty["match_number_clay"].fillna(0)
    P_pretty["match_number_grass"] = P_pretty["match_number_grass"].fillna(0)

    P_pretty = P_pretty.loc[:,["id","name","given_name","surname","alt_name","id_sackmann","country","birthday","hand","height",\
        "url","url2","surname_tennis_explorer","surname_tennis_explorer2","elo_overall","match_number","elo_hard","match_number_hard",\
        "elo_clay","match_number_clay","elo_grass","match_number_grass","previous_match"]]

    writer = pd.ExcelWriter('players.xlsx', engine='xlsxwriter',datetime_format="YYYY-MM-DD")

    P_pretty.to_excel(writer, sheet_name='players', startrow=1, header=False, index=False)

    workbook = writer.book
    worksheet = writer.sheets['players']

    cell_format = workbook.add_format({"font_size": 10, 'font_name': 'Arial'})
    m = len(P_pretty.columns)
    (max_row, max_col) = P_pretty.shape

    column_settings = [{'header': column} for column in P_pretty.columns]
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings, "style": "Table Style Medium 7"})
    worksheet.set_column(0, 0, 25, cell_format)
    worksheet.set_column(1, 1, 20, cell_format)
    worksheet.set_column(2, 3, 15, cell_format) 
    worksheet.set_column(3, 4, 20, cell_format)
    worksheet.set_column(4, 5, 8, cell_format)
    worksheet.set_column(6, 6, 25, cell_format)
    worksheet.set_column(7, 7, 10, cell_format)
    worksheet.set_column(8, 8, 6, cell_format)
    worksheet.set_column(9, 9, 8, cell_format)
    worksheet.set_column(10, 11, 65, cell_format)
    worksheet.set_column(12, 13, 15, cell_format)
    worksheet.set_column(14, 21, 10,cell_format)
    worksheet.set_column(22, 22, 100,cell_format)
    writer.close()


def elo_plot(ids_dict: dict, dates: pd.Series, tourney_dates: pd.Series, winner_ids: pd.Series, loser_ids: pd.Series, winner_elos: pd.Series, loser_elos: pd.Series, rounds: pd.Series, elo_type: str, plot_action: str):
    
    M = pd.DataFrame({"date": dates, "tourney_date": tourney_dates, "winner_id": winner_ids, "loser_id": loser_ids, "winner_elo": winner_elos, "loser_elo": loser_elos, "round": rounds})
    
    M = sort_matches_table(M,drop_date2=False)

    j = 0
    
    for name, player_id in ids_dict.items():

        M_player = M.loc[(M["winner_id"] == player_id) | (M["loser_id"] == player_id), : ].copy()
        M_player["player_elo"] = np.nan
        M_player.loc[ M_player["winner_id"] == player_id, "player_elo" ] = M_player.loc[ M_player["winner_id"] == player_id, "winner_elo"]
        M_player.loc[ M_player["winner_id"] != player_id, "player_elo" ] = M_player.loc[ M_player["winner_id"] != player_id, "loser_elo"]
        
        if elo_type == "overall":
            plt.plot(M_player["date2"], M_player["player_elo"], label = "Overall Elo for " + name)
            plt.xlabel("Year")
            plt.ylabel("Elo rating")
        elif elo_type == "hard":
            plt.plot(M_player["date2"], M_player["player_elo"], label = "Hard Court Elo for " + name)
        elif elo_type == "clay":
            plt.plot(M_player["date2"], M_player["player_elo"], label = "Clay Court Elo for " + name)
        elif elo_type == "grass":
            plt.plot(M_player["date2"], M_player["player_elo"], label = "Grass Court Elo for " + name)

        j+=1
    
    player_names = ', '.join(ids_dict.keys())
    plt.title("Elo chart of " + player_names)
    plt.legend()
    plt.grid(True)
    if plot_action == "display" or plot_action == "both":
        plt.show()
    if plot_action == "save" or plot_action == "both":
        plt.savefig("elo_chart_" + player_names + ".jpg",dpi=600)
    plt.clf()

def player_stats_table(player_table: pd.DataFrame, player_names: str, winner_ids: pd.Series, loser_ids: pd.Series, surfaces: pd.Series, winner_elos: pd.Series, loser_elos: pd.Series, winner_elos_surf: pd.Series, loser_elos_surf: pd.Series, dates: pd.Series, tourney_dates: pd.Series, rounds: pd.Series, temps: pd.Series):
    
    M = pd.DataFrame({"date": dates, "tourney_date": tourney_dates, "winner_id": winner_ids, "loser_id": loser_ids, "winner_elo": winner_elos, "loser_elo": loser_elos, "winner_elo_surface": winner_elos_surf, "loser_elo_surface": loser_elos_surf, "round": rounds, "surface": surfaces, "temp": temps})
    M = sort_matches_table(M,drop_date2=False)
    M["hour"] = M["date2"].dt.hour
    M.loc[M["hour"]==0,"hour"] = np.nan

    ids_dict = player_names_to_ids(player_table,player_names)
    row_names = ["record_overall", "record_hard","record_clay","record_grass","elo_max","elo_max_hard","elo_max_clay","elo_max_grass",\
                "current_elo","current_elo_hard","current_elo_clay","current_elo_grass","record_hot","record_cold","record_evening",\
                "record_morning","record_afternoon"]
    n = len(row_names)
    m = len(ids_dict)

    dict = {"stat": row_names}
    for name in ids_dict:
        dict[name] = [""]*n

    S = pd.DataFrame(dict)

    for name,id in ids_dict.items():
        M_player = M.loc[(M["winner_id"]==id) | (M["loser_id"]==id),:].reset_index(drop=True)
        matches_no = len(M_player.index)
        wins_no = sum(M_player["winner_id"] == id)
        losses_no = matches_no - wins_no
        wins_percent = str(round((wins_no/matches_no)*100,1))+" %"
        S.loc[0,name] = str(wins_no)+" — "+str(losses_no)+" ("+wins_percent+")" # record overall

        M_player_hard = M_player.loc[M_player["surface"] == "Hard",:].reset_index(drop=True)
        matches_hard_no = len(M_player_hard.index)
        wins_hard_no = sum(M_player_hard["winner_id"] == id)
        losses_hard_no = matches_hard_no - wins_hard_no
        wins_percent_hard = str(round((wins_hard_no/matches_hard_no)*100,1))+" %"
        S.loc[1,name] = str(wins_hard_no)+" — "+str(losses_hard_no)+" ("+wins_percent_hard+")" # record hard

        M_player_clay = M_player.loc[M_player["surface"] == "Clay",:].reset_index(drop=True)
        matches_clay_no = len(M_player_clay.index)
        wins_clay_no = sum(M_player_clay["winner_id"] == id)
        losses_clay_no = matches_clay_no - wins_clay_no
        wins_percent_clay = str(round((wins_clay_no/matches_clay_no)*100,1))+" %"
        S.loc[2,name] = str(wins_clay_no)+" — "+str(losses_clay_no)+" ("+wins_percent_clay+")" # record clay


        M_player_grass = M_player.loc[M_player["surface"] == "Grass",:].reset_index(drop=True)
        matches_grass_no = len(M_player_grass.index)
        wins_grass_no = sum(M_player_grass["winner_id"] == id)
        losses_grass_no = matches_grass_no - wins_grass_no
        if matches_grass_no > 0:
            wins_percent_grass = str(round((wins_grass_no/matches_grass_no)*100,1))+" %"
        else:
            wins_percent_grass = ""
        S.loc[3,name] = str(wins_grass_no)+" — "+str(losses_grass_no)+" ("+wins_percent_grass+")" # record grass


        M_player["player_elo"] = M_player["winner_elo"]
        M_player.loc[M["winner_id"] != id, "player_elo"] = M_player.loc[M["winner_id"] != id, "loser_elo"]
        elo_absolutemax = M_player["player_elo"].max()
        elo_max_index = np.where(M_player["player_elo"] == elo_absolutemax)[0][0]
        elo_max_date = M_player.loc[elo_max_index,"date2"].strftime("%d %b, %Y")
        S.loc[4,name] = str(round(elo_absolutemax,1))+" ("+elo_max_date+")"

        M_player_hard["player_elo_surf"] = M_player_hard["winner_elo_surface"]
        M_player_hard.loc[M["winner_id"] != id, "player_elo_surf"] = M_player_hard.loc[M["winner_id"] != id, "loser_elo_surface"]
        elo_absolutemax_hard = M_player_hard["player_elo_surf"].max()
        elo_max_hard_index = np.where(M_player_hard["player_elo_surf"] == elo_absolutemax_hard)[0][0]
        elo_max_hard_date = M_player_hard.loc[elo_max_hard_index,"date2"].strftime("%d %b, %Y")
        S.loc[5,name] = str(round(elo_absolutemax_hard,1))+" ("+elo_max_hard_date+")"

        M_player_clay["player_elo_surf"] = M_player_hard["winner_elo_surface"]
        M_player_clay.loc[M["winner_id"] != id, "player_elo_surf"] = M_player_clay.loc[M["winner_id"] != id, "loser_elo_surface"]
        elo_absolutemax_clay = M_player_clay["player_elo_surf"].max()
        elo_max_clay_index = np.where(M_player_clay["player_elo_surf"] == elo_absolutemax_clay)[0][0]
        elo_max_clay_date = M_player_clay.loc[elo_max_clay_index,"date2"].strftime("%d %b, %Y")
        S.loc[6,name] = str(round(elo_absolutemax_clay,1))+" ("+elo_max_clay_date+")"

        M_player_grass["player_elo_surf"] = M_player_grass["winner_elo_surface"]
        M_player_grass.loc[M["winner_id"] != id, "player_elo_surf"] = M_player_grass.loc[M["winner_id"] != id, "loser_elo_surface"]
        elo_absolutemax_grass = M_player_grass["player_elo_surf"].max()
        elo_max_grass_index = np.where(M_player_grass["player_elo_surf"] == elo_absolutemax_grass)[0][0]
        elo_max_grass_date = M_player_grass.loc[elo_max_grass_index,"date2"].strftime("%d %b, %Y")
        S.loc[7,name] = str(round(elo_absolutemax_grass,1))+" ("+elo_max_grass_date+")"

        current_elo = M_player.loc[(len(M_player.index)-1),"player_elo"]
        current_elo_hard = M_player_hard.loc[(len(M_player_hard.index)-1),"player_elo_surf"]
        current_elo_clay = M_player_clay.loc[(len(M_player_clay.index)-1),"player_elo_surf"]
        current_elo_grass = M_player_grass.loc[(len(M_player_grass.index)-1),"player_elo_surf"]
        current_elo_date = M_player.loc[(len(M_player.index)-1),"date2"].strftime("%d %b, %Y")
        current_elo_hard_date = M_player_hard.loc[(len(M_player_hard.index)-1),"date2"].strftime("%d %b, %Y")
        current_elo_clay_date = M_player_clay.loc[(len(M_player_clay.index)-1),"date2"].strftime("%d %b, %Y")
        current_elo_grass_date = M_player_grass.loc[(len(M_player_grass.index)-1),"date2"].strftime("%d %b, %Y")

        S.loc[8,name] = str(round(current_elo,1))+" ("+current_elo_date+")"
        S.loc[9,name] = str(round(current_elo_hard,1))+" ("+current_elo_hard_date+")"
        S.loc[10,name] = str(round(current_elo_clay,1))+" ("+current_elo_clay_date+")"
        S.loc[11,name] = str(round(current_elo_grass,1))+" ("+current_elo_grass_date+")"

        wins_hot_no = sum( (M_player["temp"] > 27) & (M_player["winner_id"] == id))
        losses_hot_no = sum( (M_player["temp"] > 27) & (M_player["winner_id"] != id))
        matches_hot_no = wins_hot_no + losses_hot_no
        wins_percent_hot = str(round((wins_hot_no/matches_hot_no)*100,1))+" %"
        S.loc[12,name] = str(wins_hot_no)+" — "+str(losses_hot_no)+" ("+wins_percent_hot+")"

        wins_cold_no = sum( (M_player["temp"] < 17) & (M_player["winner_id"] == id))
        losses_cold_no = sum( (M_player["temp"] < 17) & (M_player["winner_id"] != id))
        matches_cold_no = wins_cold_no + losses_cold_no
        wins_percent_cold = str(round((wins_cold_no/matches_cold_no)*100,1))+" %"
        S.loc[13,name] = str(wins_cold_no)+" — "+str(losses_cold_no)+" ("+wins_percent_cold+")"

        wins_evening_no = sum( (M_player["hour"] >= 18) & (M_player["winner_id"] == id))
        losses_evening_no = sum( (M_player["hour"] >= 18) & (M_player["winner_id"] != id))
        matches_evening_no = wins_evening_no + losses_evening_no
        wins_percent_evening = str(round((wins_evening_no/matches_evening_no)*100,1))+" %"
        S.loc[14,name] = str(wins_evening_no)+" — "+str(losses_evening_no)+" ("+wins_percent_evening+")"

        wins_morning_no = sum( (M_player["hour"] < 13) & (M_player["winner_id"] == id))
        losses_morning_no = sum( (M_player["hour"] < 13) & (M_player["winner_id"] != id))
        matches_morning_no = wins_morning_no + losses_morning_no
        wins_percent_morning = str(round((wins_morning_no/matches_morning_no)*100,1))+" %"
        S.loc[15,name] = str(wins_morning_no)+" — "+str(losses_morning_no)+" ("+wins_percent_morning+")"

        wins_afternoon_no = sum( (M_player["hour"].between(13,18)) & (M_player["winner_id"] == id))
        losses_afternoon_no = sum( (M_player["hour"].between(13,18)) & (M_player["winner_id"] != id))
        matches_afternoon_no = wins_afternoon_no + losses_afternoon_no
        wins_percent_afternoon = str(round((wins_afternoon_no/matches_afternoon_no)*100,1))+" %"
        S.loc[16,name] = str(wins_afternoon_no)+" — "+str(losses_afternoon_no)+" ("+wins_percent_afternoon+")"


    return S

def write_stats_table(S: pd.DataFrame):

    writer = pd.ExcelWriter("stats_"+ ", ".join(S.columns) + ".xlsx", engine='xlsxwriter')

    S.to_excel(writer, sheet_name='stats', startrow=1, header=False, index=False)

    workbook = writer.book
    worksheet = writer.sheets['stats']

    cell_format = workbook.add_format({"font_size": 10, 'font_name': 'Arial'})
    m = len(S.columns)
    (max_row, max_col) = S.shape

    column_settings = [{'header': column} for column in S.columns]
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings, "style": "Table Style Medium 7"})
    worksheet.set_column(0, 0, 20, cell_format)
    worksheet.set_column(1, (max_col-1), 20, cell_format)
    
    writer.close()


    

#print(help(update_elo))


def E_surface(name: str, E: pd.DataFrame, surface: str) -> pd.DataFrame:
    """
    Args:
        name (string): the name of the player
        E (pandas dataframe): the elos.csv table, already merged with the matches.csv table
        surface (string): the surface (either hard, clay or grass)
    
    Returns:
        the table of the matches a player played on a specific surface and the elos
    """

    E_surf = E.loc[(E.winner_name == name) | (E.loser_name == name), : ]
    E_surf = E_surf.reset_index(drop=True)
    E_surf = E_surf.loc[E_surf.surface==surface,:]
    E_surf["elo_surface"] = E_surf["loser_elo_surface"]
    E_surf.loc[E_surf.winner_name == name, "elo_surface"] = E_surf.winner_elo_surface[E_surf.winner_name == name]
    return E_surf

def E_overall(name: str, E: pd.DataFrame):
    """
    Args:
        name (string): the name of the player
        E (pandas dataframe): the elos.csv table, already merged with the matches.csv table
    
    Returns:
        the table all of the matches the player played including the elos
    """

    E_over = E.loc[(E.winner_name == name) | (E.loser_name == name), : ]
    E_over = E_over.reset_index(drop=True)
    E_over["elo_overall"] = E_over.loser_elo
    E_over.loc[E_over.winner_name == name, "elo_overall"] = E_over.winner_elo[E_over.winner_name == name]
    return E_over

def weather_data(googledrive_url,M):
    W = pd.read_csv("https://drive.google.com/uc?id=" + googledrive_url.split("/")[-2])
    W.date = pd.to_datetime(W.date.astype("string")+W.hour.astype("string")+"00", format = '%Y%m%d%H%M%S')
    W = W.sort_values(by = ["date"], ascending = True)
    W = W.reset_index(drop=True)
    r = requests.get(googledrive_url)
    tree = fromstring(r.content)
    title = tree.findtext('.//title')
    title = title[:(title.find("Google Drive")-7)]
    weather_url = "https://rp5.ru/Weather_archive_in_" + title
    M2 = M.loc[M.URL==weather_url,:]
    M2.loc[:,"hour"] = M2.loc[:,"date"].dt.hour
    M2 = M2.loc[M2.hour!=0,:]
    M2 = M2.drop(["hour"],axis=1)
    M2 = M2.reset_index(drop=True)
    M2["timestamp_before"] = datetime.strptime("20091201000000", '%Y%m%d%H%M%S')
    M2["timestamp_after"] = datetime.strptime("20240101000000", '%Y%m%d%H%M%S')
    for i in range(len(M2.index)-1):
        row_after = W["date"].gt(M2.date[i]).idxmax() # first instance of the variable W.date being >= M2.date[i]
        M2.loc[i,"timestamp_before"] = W.date[(row_after-1)]
        M2.loc[i,"timestamp_after"] = W.date[row_after]

    W = W.drop(["hour"],axis=1)
    W_before = W.rename({"date": "timestamp_before", "temp_c": "temp_c_before", "hum_perc": "hum_perc_before", \
        "pressure_mmHg": "pressure_mmHg_before", "windspeed_ms": "windspeed_ms_before"}, axis=1)
    M2 = pd.merge(M2,W_before,how="left",on="timestamp_before")
    W_after = W.rename({"date": "timestamp_after", "temp_c": "temp_c_after", "hum_perc": "hum_perc_after", \
        "pressure_mmHg": "pressure_mmHg_after", "windspeed_ms": "windspeed_ms_after"}, axis=1)
    M2 = pd.merge(M2,W_after,how="left",on="timestamp_after")
    M2["minutes_before"] = (M2.date-M2.timestamp_before).dt.total_seconds()
    M2["minutes_after"] = (M2.timestamp_after-M2.date).dt.total_seconds()
    # The temperature at the actual time will be estimated by calculating the weighted average between temp_before and temp_after
    # We assume a linear graph, i.e. the weight for temp_before is (minutes_after)/(minutes_before+minutes_after) and the weight
    # for temp_after is (minutes_before)/(minutes_before+minutes_after)
    M2["temp_c"] = round(M2.minutes_after/(M2.minutes_before+M2.minutes_after)*M2.temp_c_before + \
        M2.minutes_before/(M2.minutes_before+M2.minutes_after)*M2.temp_c_after,1)
    M2["hum_perc"] = round(M2.minutes_after/(M2.minutes_before+M2.minutes_after)*M2.hum_perc_before + \
        M2.minutes_before/(M2.minutes_before+M2.minutes_after)*M2.hum_perc_after,1)
    M2["pressure_mmHg"] = round(M2.minutes_after/(M2.minutes_before+M2.minutes_after)*M2.pressure_mmHg_before + \
        M2.minutes_before/(M2.minutes_before+M2.minutes_after)*M2.pressure_mmHg_after,1)
    M2["windpssed_ms"] = round((M2.minutes_after/(M2.minutes_before+M2.minutes_after)*M2.windspeed_ms_before + \
        M2.minutes_before/(M2.minutes_before+M2.minutes_after)*M2.windspeed_ms_after)*2)/2
    M2 = M2.drop(["URL","date","timestamp_before","timestamp_after","minutes_before","minutes_after","temp_c_before","temp_c_after",\
        "hum_perc_before","hum_perc_after","pressure_mmHg_before","pressure_mmHg_after","windspeed_ms_before","windspeed_ms_after"],axis=1)
    return M2

def strip_strings(x):
    if isinstance(x, str):
        return x.strip()
    else:
        return x

def download_from_github(year: int = datetime.now().year, matches_table: pd.DataFrame = read_matches(), players_table: pd.DataFrame = read_players(), tournaments_table: pd.DataFrame = read_tournaments()):
    
    matches_github_columns = ["tourney_id","tourney_name","surface","tourney_date","score","best_of","round","minutes","w_ace","w_df","w_svpt","w_1stIn","w_1stWon",\
                              "w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced","l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced",\
                                "winner_name","loser_name"]

    matches_github_dtypes = {"tourney_id": "str", "tourney_name": "str", "tourney_date": "str", "score": "str", "best_of": "Int64", "round": "str", "minutes": "Int64", \
                             "w_ace": "Int64", "w_df": "Int64", "w_svpt": "Int64", "w_1st": "Int64", "w_1stWon": "Int64", "w_2ndWon": "Int64", "w_SvGms": "Int64", \
                             "w_bpSaved": "Int64", "w_bpFaced": "Int64", "l_ace": "Int64", "l_df": "Int64", "l_svpt": "Int64", "l_1st": "Int64", "l_1stWon": "Int64", \
                             "l_2ndWon": "Int64", "l_SvGms": "Int64", "l_bpSaved": "Int64", "l_bpFaced": "Int64" }
    
    players_github_dtypes = {"player_id": "Int64", "name_first": "str", "name_last": "str", "hand": "str", "dob": "Int64", "ioc": "str", "height": "Int64", "wikidata_id": "str"}
    
    
    matches_atp = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_" + str(year) + ".csv", dtype = matches_github_dtypes, usecols = matches_github_columns, parse_dates=["tourney_date"])
    matches_chal = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_qual_chall_" + str(year) + ".csv", dtype = matches_github_dtypes, usecols = matches_github_columns, parse_dates=["tourney_date"])
    matches_itf = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_" + str(year) + ".csv", dtype = matches_github_dtypes, usecols = matches_github_columns, parse_dates=["tourney_date"])
    players_github = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv", dtype = players_github_dtypes)
    ioc_codes = pd.read_csv("ioc_codes.csv")

    matches_github = pd.concat([matches_atp,matches_chal,matches_itf]).reset_index(drop=True)

    matches_github.to_csv("matches_new_0.csv",index=False)

    # Strip blanks
    matches_github = matches_github.applymap(strip_strings)
    matches_github = matches_github.applymap(lambda x: x.replace("  ", " ") if isinstance(x, str) else x)
    players_github = players_github.applymap(strip_strings)

    matches_github["year"] = year
    matches_github["tourney_id"] = tourney_id(matches_github["year"], matches_github["tourney_name"], matches_github["tourney_date"])

    alt_names = players_table.loc[~players_table.alt_name.isna(),["name","alt_name"]].reset_index(drop=True)
    for row in alt_names.itertuples():
        matches_github.loc[ matches_github["winner_name"] == row.alt_name, "winner_name" ] = row.name
        matches_github.loc[ matches_github["loser_name"] == row.alt_name, "loser_name" ] = row.name
    

    players_github["name"] = players_github["name_first"] + " " + players_github["name_last"]
    player_not_available = (~players_github["name"].isin(players_table["name"])) & (~players_github["name"].isin(players_table["alt_name"]))
    player_relevant = players_github["name"].isin(matches_github["winner_name"]) | players_github["name"].isin(matches_github["loser_name"])
    players_new = players_github.loc[player_not_available & player_relevant, : ]
    players_new = players_new.rename({"player_id": "id_sackmann"}, axis = 1)
    players_new = pd.merge(players_new, ioc_codes, how = "left", left_on = "ioc", right_on = "code")
    players_new["birthday"] = pd.to_datetime(players_new["dob"], format = "%Y%m%d",errors="coerce")
    players_new["hand"] = np.where(players_new["hand"] == "L", "left", np.where(players_new["hand"] == "R", "right", np.nan))
    players_new["id"] = player_id(players_new["name"], players_new["country"], players_new["birthday"])
    players_new["elo_overall"] = 1400
    players_new["elo_hard"] = 1400
    players_new["elo_clay"] = 1400
    players_new["elo_grass"] = 1400
    players_new["match_number"] = 0
    players_new["match_number_hard"] = 0
    players_new["match_number_clay"] = 0
    players_new["match_number_grass"] = 0
    

    players_total = pd.concat([players_table,players_new]).reset_index(drop=True)

    players_lookup_winner = players_total.loc[:,["name","id"]]
    players_lookup_winner = players_lookup_winner.rename({"name": "winner_name", "id": "winner_id"},axis=1)
    matches_github = pd.merge(matches_github, players_lookup_winner, how = "left", on = "winner_name")
    players_lookup_loser = players_total.loc[:,["name","id"]]
    players_lookup_loser = players_lookup_loser.rename({"name": "loser_name", "id": "loser_id"},axis=1)
    matches_github = pd.merge(matches_github, players_lookup_loser, how = "left", on = "loser_name")
    matches_github["match_id"] = match_id(matches_github["tourney_id"], matches_github["winner_id"], matches_github["loser_id"], matches_github["score"])
    matches_not_available = ~(matches_github["match_id"].isin(matches_table["match_id"]))
    matches_new = matches_github.loc[matches_not_available, : ]

    #players_total_winner = players_total.loc[:,["name","id"]].rename({"name": "winner_name", "id": "winner_id"},axis=1)
    #matches_new = pd.merge(matches_new,players_total_winner,how="left",on="winner_name")


    players_new = players_new.drop(["name_first","name_last","wikidata_id","dob","ioc","code"], axis = 1)


    tournaments_github = matches_github.loc[:,["tourney_id","tourney_name","surface","tourney_date"]]
    tournaments_github["year"] = year
    tournaments_github = tournaments_github.drop_duplicates().reset_index(drop=True)
    tournaments_github = tournaments_github.rename({"tourney_date": "date", "tourney_name": "name", "tourney_id": "id"},axis=1)
    tournament_not_available = (~tournaments_github["id"].isin(tournaments_table["id"]))
    tournaments_new = tournaments_github.loc[tournament_not_available, : ]

    #M = M.drop(["tourney_name", "surface","tourney_date"], axis = 1)

    #matches_new = matches_new.drop(["tourney_name","surface","tourney_date","tourney_level","match_num","winner_seed","winner_entry","winner_ht","winner_ioc",\
    #                                "winner_age","loser_seed","loser_entry","loser_name","loser_ht","loser_ioc","loser_age","winner_rank","winner_rank_points","loser_rank",\
    #                                "loser_rank_points","year","winner_id_y","loser_id_y","match_id","winner_hand","loser_hand"], axis = 1)

    #col = ["match_id","year","surface","tourney_date","tourney_name"]
    #for c in col:
    #    matches_new = matches_new.drop([c],index=1)
    #    if c in matches_new.columns:
    #        print("{} in columns".format(c))
    matches_new = matches_new.drop(["year","surface","tourney_date","tourney_name","winner_name","loser_name"],axis=1)

    return matches_new, tournaments_new, players_new


def aces_per_point(M: pd.DataFrame, year: int, tourney_name: str) -> int:
    tourney_id = str(year) + "-" + tourney_name
    print(len(M.index))
    print(sum(M.tourney_id==tourney_id))
    M = M.loc[M.tourney_id == tourney_id,:]
    #M = M.reset_index(drop=True)
    M.to_csv("US Open.csv",index=False)
    print(M)
    print(M.w_ace)
    total_aces = sum(M.w_ace) +  sum(M.l_ace)
    print(total_aces)
    total_points = sum(M.w_svpt) +  sum(M.l_svpt)
    print(total_points)
    return total_aces/total_points



