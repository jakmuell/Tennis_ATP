import unittest
from tennis_functions import *
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="The default value of regex will change from True to False in a future version.")

big3_ids = ["rafael nadal-90","roger federer-sw-85","novak djokovic"]
big3_birthdays = [datetime(1990,5,25),datetime(1985,1,1), np.nan]
big3_elos = [2000,2100,1950]
big3_match_number = [0,0,0]
big3_df = pd.DataFrame({"id": big3_ids, "birthday": big3_birthdays, "elo_overall": big3_elos, "match_number": big3_match_number})

class TestPlayerId(unittest.TestCase):
    def test_playerid(self):
        names = ["Rafael Nadal","Roger Federer","Novak Djokovic"]
        birthdays = [datetime(1990,5,25),datetime(1985,1,1), np.nan]
        countries = [np.nan,"Switzerland",np.nan]
        P = pd.DataFrame({"name": names, "birthday": birthdays, "country": countries})
        player_ids_return = player_id(P.name,P.country,P.birthday)
        player_ids_expected = np.array(["rafael nadal-90","roger federer-sw-85","novak djokovic"])
        np.testing.assert_array_equal(player_ids_return,player_ids_expected)

class TestTourneyId(unittest.TestCase):
    def test_tourneyid(self):
        names = ["Australian Open","US Open","Wimbledon"]
        years = [2010,2011,2012]
        dates = [datetime(2010,1,15,0,0),datetime(2011,9,1,0,0),np.nan]
        T = pd.DataFrame({"name": names, "year": years, "date": dates})
        names = T.name
        years = T.year
        dates = T.date
        tourney_ids_return = tourney_id(T.year,T.name,T.date)
        tourney_ids_expected = np.array(["2010-australian open-20100115","2011-us open-20110901","2012-wimbledon"])
        np.testing.assert_array_equal(tourney_ids_return,tourney_ids_expected)

class TestPlayerIndex(unittest.TestCase):
    def test_playerindex(self):
        winner_ids = np.array(["rafael nadal-90","roger federer-sw-85","novak djokovic"])
        loser_ids = np.array(["roger federer-sw-85","rafael nadal-90","roger federer-sw-85"])
        ids_lookup = ["rafael nadal-90","roger federer-sw-85","novak djokovic"]
        winner_rows_expected = np.array([0,1,2])
        loser_rows_expected = np.array([1,0,1])
        winner_rows_returned, loser_rows_returned = winner_and_loser_row(winner_ids,loser_ids,ids_lookup)
        np.testing.assert_array_equal(winner_rows_returned,winner_rows_expected)
        np.testing.assert_array_equal(loser_rows_returned,loser_rows_expected)



class TestRetrievefromP(unittest.TestCase):
    def test_retrievefromP_nadal_djokovic(self):
        P = read_players()
        winner_row = np.where(P["name"] == "Rafael Nadal")[0][0]
        loser_row = np.where(P["name"] == "Novak Djokovic")[0][0]
        w_elo_overall, l_elo_overall, w_elo_surface, l_elo_surface, w_match_number, l_match_number, w_match_number_surface, l_match_number_surface = retrieve_data_from_P(P,winner_row,loser_row,"Hard")
        w_elo_overall_expected = P.elo_overall[P.name == "Rafael Nadal"].item()
        l_match_number_surface_expected = P.match_number_hard[P.name == "Novak Djokovic"].item()
        self.assertEqual(w_elo_overall,w_elo_overall_expected)
        self.assertEqual(l_match_number_surface,l_match_number_surface_expected)

class TestEloFactors(unittest.TestCase):
    
    def test_elofactors_youngplayer(self):
        activityfactor_returned, penaltyfactor_returned = elo_factors(1700,5,2)
        activityfactor_expected, penaltyfactor_expected = 1, 1
        self.assertEqual(activityfactor_returned,activityfactor_expected)
        self.assertEqual(penaltyfactor_returned,penaltyfactor_expected)

    def test_elofactors_experiencedplayer(self):
        activityfactor_returned, penaltyfactor_returned = elo_factors(2200,600,20)
        activityfactor_expected, penaltyfactor_expected = exp(-0.4*20)+1, 1
        self.assertEqual(activityfactor_returned,activityfactor_expected)
        self.assertEqual(penaltyfactor_returned,penaltyfactor_expected)

    def test_elofactors_injuredplayer(self):
        activityfactor_returned, penaltyfactor_returned = elo_factors(2100,400,0)
        activityfactor_expected, penaltyfactor_expected = 2, 1-(1-0.98)/(1+exp(-0.05*(2100-1910))*((1/0.995)-1))
        self.assertEqual(activityfactor_returned,activityfactor_expected)
        self.assertEqual(penaltyfactor_returned,penaltyfactor_expected)
        self.assertLess(penaltyfactor_returned,1)

class TestEloNew(unittest.TestCase):
    def test_elo_new(self):
        K_djok = 40
        K_fed = 30
        djok_elo = 1950
        fed_elo = 2100
        djok_setwinprob = 1/ (1 + 10**((fed_elo - djok_elo)/400))
        djok_elogain_by_set = K_djok * (1-djok_setwinprob)
        fed_eloloss_by_set = K_fed * (1-djok_setwinprob)
        djok_elo_new_exp = 1950+2*djok_elogain_by_set
        fed_elo_new_exp = 2100-2*fed_eloloss_by_set
        djok_elo_new_return, fed_elo_new_return = elo_new(djok_elo,fed_elo,K_djok,K_fed,2,0)
        self.assertAlmostEqual(djok_elo_new_return,djok_elo_new_exp)
        self.assertAlmostEqual(fed_elo_new_return,fed_elo_new_exp)
    

class TestUpdateElo(unittest.TestCase):

    def test_update_elo_basic(self):
        winner_ids = ["novak djokovic","rafael nadal-90","novak djokovic","rafael nadal-90","roger federer-sw-85"]
        loser_ids = ["roger federer-sw-85","roger federer-sw-85","rafael nadal-90","novak djokovic","rafael nadal-90"]
        tourney_ids = ["id1","id2","id3","id4","id5"]
        surfaces = ["Hard","Clay","Hard","Clay","Hard"]
        levels = ["ATP 500","ATP 250","ATP 1000","ATP 1000","Grand Slam"]
        scores = ["6-1 7-6(5)","6-4 3-6 6-4","7-6(5) 6-7(3) 6-4","6-4 6-4","6-4 3-6 1-6 7-6(5) 7-5"]
        dates = [datetime(2010,1,1,20,0),datetime(2010,1,10,18,0),datetime(2010,2,1,12,0),datetime(2010,2,2,12,0),datetime(2010,3,1,21,0)]
        M_old = pd.DataFrame({"winner_id": winner_ids, "loser_id": loser_ids, "score": scores, "surface": surfaces, "tourney_level": levels, "tourney_id": tourney_ids, "date": dates})
        M_old["match_id"] = match_id(M_old.tourney_id,M_old.winner_id,M_old.loser_id,M_old.score)

        winner_setswon_tmp, loser_setswon_tmp = sets_won_by_player(M_old.score)
        M_old["winner_setswon"] = winner_setswon_tmp
        M_old["loser_setswon"] = loser_setswon_tmp
        M, P, elapsed = update_elo(M_old,big3_df,0,len(M_old.index)-1)
        M.to_csv("M_test.csv")
        P.to_csv("P_test.csv")
        

        K_overall = K_factor(0)
        K_hard = K_factor(0,280)
        djok_setwinprob = 1/ (1 + 10**((2100-1950)/400))
        djok_elogain_by_set = K_overall * (1-djok_setwinprob)
        djok_elogain_by_set_hard = K_hard * (1-djok_setwinprob)
        fed_eloloss_by_set = K_overall * (1-djok_setwinprob)
        fed_eloss_by_set_hard = K_hard * (1-djok_setwinprob)
        djok_elo_new_exp = 1950+2*djok_elogain_by_set
        fed_elo_new_exp = 2100-2*fed_eloloss_by_set
        djok_elo_new_hard_exp = 1950+2*djok_elogain_by_set_hard
        fed_elo_new_hard_exp = 2100-2*fed_eloss_by_set_hard
        djok_elo_new_return, fed_elo_new_return = M.winner_elo[0], M.loser_elo[0]
        djok_elo_new_hard_return, fed_elo_new_hard_return = M.winner_elo_surface[0], M.loser_elo_surface[0]
        self.assertAlmostEqual(djok_elo_new_hard_return,djok_elo_new_hard_exp)
        self.assertAlmostEqual(djok_elo_new_return,djok_elo_new_exp)
        self.assertAlmostEqual(fed_elo_new_hard_return,fed_elo_new_hard_exp)
        self.assertAlmostEqual(fed_elo_new_return,fed_elo_new_exp)


class TestSetwinner(unittest.TestCase):

    def test_setwinner_basic(self):
        sets = pd.Series(["6-3","0-6","7-6(4)","[10-4]","18-16","6-7","2-1","RET"])
        expected_result = np.array(["winner", "loser", "winner", "winner", "winner", "loser", "neither", "neither"])
        np.testing.assert_array_equal(setwinner(sets),expected_result)

    #def test_setwinner_empty(self):
    #    sets = pd.Series([])
    #    expected_result = np.array([])
    #    self.assertEqual(setwinner(sets),expected_result)


class TestSetswonbyplayer(unittest.TestCase):

    def test_setswonbyplayer_basic(self):
        scores = pd.Series(["6-3 6-4", "6-0 0-6 7-6(1)", "6-1 1-6 2-0 RET", "6-0 6-0 0-6 0-6 6-4", "W/O"])
        winner_setswon_expected = np.array([2,2,1,3,0])
        loser_setswon_expected = np.array([0,1,1,2,0])
        winner_setswon_return, loser_setswon_return = sets_won_by_player(scores)
        np.testing.assert_array_equal(winner_setswon_return,winner_setswon_expected)
        np.testing.assert_array_equal(loser_setswon_return,loser_setswon_expected)

if __name__ == "__main__":
    unittest.main()
