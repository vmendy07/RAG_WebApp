import requests

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/"

def get_players():
    response = requests.get(f"{BASE_URL}players")
    return response.json()

def get_player_stats():
    response = requests.get(f"{BASE_URL}statistics/players")
    return response.json()

def get_team_roster(team):
    response = requests.get(f"{BASE_URL}{team}/roster")
    return response.json()

def get_team_schedule(team):
    response = requests.get(f"{BASE_URL}{team}/schedule")
    return response.json()
