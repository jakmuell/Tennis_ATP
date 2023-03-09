print("Importing modules, this will take just a couple of seconds.")

from datetime import datetime
import pandas as pd
import os
import warnings
from tennis_functions import *

warnings.filterwarnings("ignore", message="The default value of regex will change from True to False in a future version.")

# Change working directory directory where the file is located
abspath = os.path.abspath(__file__)
maindirectory = os.path.dirname(abspath)
os.chdir(maindirectory)

bool_exit_main_menu = False

bool_players_start_table_loaded = False

print("Reading all the datasets, this will take just a couple of seconds.")
matches_table = read_matches()
tournaments_table = read_tournaments()
cities_table = read_cities()
players_table = read_players()
master_table = create_master_table(True,True,matches_table,tournaments_table,cities_table)


def generate_player_name_id_dict():
    print('Type the name of one or more players (e.g. type \"Rafael Nadal, Novak Djokovic, Roger Federer\" or simply type \"Rafael Nadal\").')
    player_names = input()
    names_list = player_names.split(", ")
    ids_list = ["tmp"]*len(names_list)
    dict = {}
    for name in names_list:
        tmp = players_table.loc[players_table["name"] == name, "id"]
        while len(tmp) == 0:
            print("There is no player named {}. Please correct your input for this player.".format(name))
            name = input()
            tmp = players_table.loc[players_table["name"] == name, "id"]
        dict[name] = tmp.item()

    return dict

def create_subfolder(foldername: str):
    if os.path.isdir(foldername):
        os.chdir(foldername)
    else:
        os.mkdir(foldername)
        os.chdir(foldername)


subfoldername = datetime.now().strftime("%Y%m%d_%H%M%S")







while not bool_exit_main_menu:
    print("What do you want to do?")
    print("[1] Calculate elos.")
    print("[2] Calculate player stats.")
    print("[3] Check for update for matches on https://github.com/JeffSackmann/tennis_atp.")
    print("[4] Check for updates in the folder \"scores\".")
    print("[5] Exit.")
    main_menu = int(input())
    if main_menu == 1:
        if not bool_players_start_table_loaded: # Import elo_ratings_yearend_2009, but only if we haven't done so already
            P_start_dtypes = {"id": "str", "elo_overall": "float", "match_number": "int"}
            P_start = pd.read_csv("elo_ratings_yearend_2009.csv", dtype = P_start_dtypes )
            
        elos_match_by_match, players_ratings_final = update_elo(master_table["match_id"],master_table["winner_id"],master_table["loser_id"],master_table["score"],master_table["tourney_date"],\
                        master_table["date"],master_table["surface"],master_table["tourney_level"],master_table["round"],P_start)
        bool_exit_elo_menu = False
        while not bool_exit_elo_menu:
            print("What do you want to do?")
            print("[1] Write the two resulting tables to csv files.")
            print("[2] Write to the files \"matches_10_15.csv\", \"matches_16_end.csv\" and \"players.xlsx\" with the new information in a subfolder.")
            print("[3] Write to the files \"matches_10_15.csv\", \"matches_16_end.csv\" and \"players.xlsx\" with the new information in the main folder.")
            print("[4] Exit.")
            try:
                elo_menu = int(input())
            except:
                elo_menu = "Invalid"
                print("Invalid input. Try again")
            if elo_menu == 1:
                create_subfolder(subfoldername)
                elos_match_by_match.to_csv("elos_match_by_match.csv",index=False)
                players_ratings_final.to_csv("players_elos.csv",index=False)
                os.chdir(maindirectory)
            elif elo_menu == 2 or elo_menu == 3:
                players_info_table = players_table.loc[:,["id","name","given_name","surname","alt_name","id_sackmann","country","birthday"\
                                                          ,"hand","height","url","url2","surname_tennis_explorer","surname_tennis_explorer2",]]
                players_table_updated = pd.merge(players_info_table, players_ratings_final, how = "left", on = "id")
                #players_table_updated.to_csv("players_table_updates.csv",index=False)
                matches_table_columns_not_update = ["match_id","winner_id", "loser_id", "score", "best_of", "round", "minutes", "w_ace", "w_df", "w_svpt",	"w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved",\
                                                    "w_bpFaced", "l_ace", "l_df", "l_svpt",	"l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", \
                                                    "l_bpFaced", "date", "temp", "wind", "hum",	"tourney_id"]
                #matches_table_columns_do_update = ["match_id", "winner_elo", "loser_elo", "winner_elo_surface", "loser_elo_surface", \
                #                                   "winner_previous_match", "loser_previous_match"]
                matches_table_updated = pd.merge(left = matches_table.loc[:,matches_table_columns_not_update], right = elos_match_by_match, on = "match_id", how = "left")
                matches_table_updated = matches_table_updated.drop(["match_id"], axis=1)
                old_columns = set(matches_table.columns).difference({"match_id"})
                new_columns = set(matches_table_updated.columns)
                if len(matches_table_updated.index) != len(matches_table.index):
                    warnings.warn("The new matches table does not have the same number of rows as the old one, something probably went wrong.", UserWarning, stacklevel=2)
                if len(new_columns) > len(old_columns):
                    setdiff = new_columns.difference(old_columns)
                    warnings.warn("The new matches table has more columns than the old one. The following columns are new: {}.".format(setdiff), UserWarning, stacklevel=2)
                if len(new_columns) < len(old_columns):
                    setdiff = old_columns.difference(new_columns)
                    warnings.warn("The new matches table has fewer columns than the old one ({} compared to {}). The following columns are missing: {}.".format(len(matches_table_updated.columns),len(matches_table.columns),setdiff), UserWarning, stacklevel=2)
                matches_table_updated1 = matches_table_updated.loc[tourney_id_to_year(matches_table["tourney_id"])<2016]
                matches_table_updated2 = matches_table_updated.loc[tourney_id_to_year(matches_table["tourney_id"])>=2016]
                if elo_menu == 2:
                    create_subfolder(subfoldername)
                    matches_table_updated1.to_csv("matches_10_15.csv",index=False)
                    matches_table_updated2.to_csv("matches_16_end.csv",index=False)
                    write_player_table(players_table_updated)
                    os.chdir(maindirectory)
                elif elo_menu == 3:
                    print("\033[31mWarning: This will permanently change the already existing files. Only continue if you have backup of the files. Continue? [Y/N]\033[0m")
                    continue_bool = input()
                    if continue_bool == "Y":
                        matches_table_updated1.to_csv("matches_10_15.csv",index=False)
                        matches_table_updated2.to_csv("matches_16_end.csv",index=False)
                        write_player_table(players_table_updated)
            elif elo_menu == 4:
                bool_exit_elo_menu = True
                continue
    
    elif main_menu == 2:
            
        bool_exit_playerstats_menu = False
        while not bool_exit_playerstats_menu:
            print("What do you want to do?")
            print("[1] Create table of all matches a player played.")
            print("[2] Create plot of a player's Elo rating over time.")
            print("[3] Compare stats of two players in a table.")

            option = int(input())
            if option == 1:
                print("Type a player name.")
                name = input()
                id = players_table.loc[players_table["name"] == name, "id"].item()
                M_player = master_table.loc[(master_table["winner_id"] == id) | (master_table["loser_id"] == id), : ].copy()
                M_player["won/lost"] = "lost"
                M_player.loc[M_player["winner_id"] == id, "won/lost"] = "won"
                M_player["opponent"] = M_player["winner_id"]
                M_player.loc[M_player["winner_id"] == id, "opponent"] = M_player.loc[M_player["winner_id"]==id,"loser_id"]
                M_player = sort_matches_table(M_player)
                M_player = M_player.drop(["winner_id","loser_id"], axis=1)
                M_player.to_csv("matches_" + name + ".csv", index = False)

            if option == 2:
                players_dict = generate_player_name_id_dict()
                print("Which Elo type are you interested in?")
                print("[1] Overall Elo")
                print("[2] Hard court Elo")
                print("[3] Clay court Elo")
                print("[4] Grass court Elo")
                elo_type_int = int(input())
                elo_type = "tmp"
                if elo_type_int == 1:
                    elo_type = "overall"
                elif elo_type_int == 2:
                    elo_type = "hard"
                elif elo_type_int == 3:
                    elo_type = "clay"
                elif elo_type_int == 4:
                    elo_type = "grass"
                print("What do you want to do with the plot?")
                print("[1] Display")
                print("[2] Save")
                print("[3] Display and save")
                plot_action_int = int(input())
                plot_action = "tmp"
                if elo_type_int == 1:
                    plot_action = "display"
                elif elo_type_int == 2:
                    plot_action = "save"
                elif elo_type_int == 3:
                    plot_action = "both"

                elo_plot(players_dict, master_table["date"],master_table["tourney_date"],master_table["winner_id"],master_table["loser_id"],master_table["winner_elo"],master_table["loser_elo"],master_table["round"],elo_type,plot_action)
            if option==3:
                # https://stackoverflow.com/questions/65761938/matplotlib-table-size-and-position
                print('Type the name of one or more players (e.g. type \"Rafael Nadal, Novak Djokovic, Roger Federer\" or simply type \"Rafael Nadal\").')
                player_names = input()
                S = player_stats_table(players_table,player_names,master_table["winner_id"],master_table["loser_id"],master_table["surface"],\
                                       master_table["winner_elo"],master_table["loser_elo"],master_table["winner_elo_surface"],master_table["loser_elo_surface"],\
                                        master_table["date"],master_table["tourney_date"],master_table["round"],master_table["temp"])
                print("What do you want to do with the table?")
                print("[1] Print to screen.")
                print("[2] Export to Excel.")
                stats_option = int(input())
                if stats_option == 1:
                    print(S)
                elif stats_option == 2:
                    #S.to_csv("stats_" + player_names + ".csv",encoding="cp1252",index=False)
                    write_stats_table(S)

    elif main_menu == 3:
        matches_new, tournaments_new, players_new = download_from_github()
        if len(matches_new.index) == 0 and len(tournaments_new.index==0) and len(players_new.index==0):
            print("No new data found.")
        else:
            print("New data found. What do you want to do with the new data?")
            print("[1] Write to the tables \"matches_new.csv\", \"tournaments_new.csv\", \"players_new.csv\".")
            print("[2] Write to the tables \"matches_16_end.csv\", \"tournaments.xlsx\" and \"players.xlsx\" in a subfolder.")
            print("[3] Write to the tables \"matches_16_end.csv\", \"tournaments.xlsx\" and \"players.xlsx\" in the main folder.")
            print("[4] Exit.")
            github_update_menu = int(input())

            if github_update_menu == 1:
                create_subfolder(subfoldername)
                matches_new = matches_new.drop(["match_id"],axis=1)
                matches_new.to_csv("matches_new.csv",index=False)
                tournaments_new.to_csv("tournaments_new.csv",index=False)
                players_new.to_csv("players_new.csv",index=False)
                os.chdir(maindirectory)
            
            elif github_update_menu == 2 or github_update_menu == 3:
                matches_16_end_before = matches_table.loc[tourney_id_to_year(matches_table["tourney_id"])>=2016,:]
                matches_16_end_total = pd.concat([matches_16_end_before,matches_new],join="outer",ignore_index=True)
                matches_16_end_total = matches_16_end_total.drop(["match_id"],axis=1)
                
                players_total = pd.concat([players_table,players_new],join="outer",ignore_index=True)
                tournaments_total = pd.concat([tournaments_table,tournaments_new],join="outer",ignore_index=True)

                if github_update_menu == 2:
                    create_subfolder(subfoldername)
                    matches_16_end_total.to_csv("matches_16_end.csv",index=False)
                    write_player_table(players_total)
                    write_tournaments_table(tournaments_total)
                    os.chdir(maindirectory)

                if github_update_menu == 3:
                    print("\033[31mWarning: This will permanently change the already existing files. Only continue if you have backup of the files. Continue? [Y/N]\033[0m")
                    continue_bool = input()
                    if continue_bool == "Y":
                        matches_16_end_total.to_csv("matches_16_end.csv",index=False)
                        write_player_table(players_total)
                        write_tournaments_table(tournaments_total)



    elif main_menu == 4:
        os.chdir("scores")
        matches_table = matches_table.loc[master_table["tourney_level"]!="European League System",:]
        tournaments_table = tournaments_table.loc[tournaments_table["level"]!="European League System",:]
        
        overview_dtypes = {"country": "str", "surface": "str", "outdoor": "bool", "filename": "str"}
        df = pd.read_csv("overview.csv",dtype=overview_dtypes)
        for index, row in df.iterrows():
            M, T = european_league_scores(row["country"],row["surface"],row["outdoor_bool"],players_table,row["filename"])
            matches_table = pd.concat([matches_table,M]).reset_index(drop=True)
            tournaments_table = pd.concat([tournaments_table,T]).reset_index(drop=True)

        memory_usage = matches_table.memory_usage(deep=True).sum()
        print(f"Memory usage of df1: {memory_usage / (1024 * 1024):.2f} MB")
        print(sum(matches_table["winner_previous_match"]==""))
        print(sum(matches_table["winner_previous_match"].isnull()))
        #del master_table
        #master_table = create_master_table(True,True,matches_table,tournaments_table,cities_table) # Update master table in case we want to do something with it later
        
        print("\033[31mWarning: This will permanently change the files \"tournaments.xlsx\", \"matches_10_15.csv\" and \"matches_16_end.csv\". Only continue if you have backup of the files. Continue? [Y/N]\033[0m")
        continue_bool = input()
        if continue_bool == "Y":
            os.chdir(maindirectory)
            M1 = matches_table.loc[tourney_id_to_year(matches_table["tourney_id"])<2016,:]
            M2 = matches_table.loc[tourney_id_to_year(matches_table["tourney_id"])>=2016,:]
            M1.to_csv("matches_10_15.csv",index=False)
            M2.to_csv("matches_16_end.csv",index=False)
            write_tournaments_table(tournaments_table)
        
    
    elif main_menu == 5:
        bool_exit_main_menu = True
        continue