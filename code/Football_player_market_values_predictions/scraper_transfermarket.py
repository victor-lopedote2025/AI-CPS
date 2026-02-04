from anyio import value
import pandas as pd
import cloudscraper
import bs4 as bs 
import time
import glob
import os

URL_ROOT = "https://www.transfermarkt.com/"
URL = ["https://www.transfermarkt.com/premier-league/marktwerteverein/wettbewerb/GB1", 
       "https://www.transfermarkt.com/ligue-1/marktwerteverein/wettbewerb/FR1",
       "https://www.transfermarkt.com/laliga/marktwerteverein/wettbewerb/ES1",
       "https://www.transfermarkt.com/bundesliga/marktwerteverein/wettbewerb/L1",
       "https://www.transfermarkt.com/serie-a/marktwerteverein/wettbewerb/IT1",
       "https://www.transfermarkt.com/jupiler-pro-league/marktwerteverein/wettbewerb/BE1",
       "https://www.transfermarkt.com/eredivisie/marktwerteverein/wettbewerb/NL1",
       "https://www.transfermarkt.com/super-lig/marktwerteverein/wettbewerb/TR1",
       "https://www.transfermarkt.com/saudi-pro-league/marktwerteverein/wettbewerb/SA1",
       "https://www.transfermarkt.com/mlstm/marktwerteverein/wettbewerb/MLS1"
       ]

def get_transfer_market_data_player_stats_for_competition(player_market_stats, name):

    NATIONAL_COMPETITION_NAMES = ['Premier League', 'Serie A', 'Ligue 1', 'Bundesliga', 'LaLiga', 'Jupiler Pro League',
                                  'Eredivisie', 'Süper Lig', 'Saudi Pro League', 'MLS']
    INTERNATIONAL_COMPETITION_NAMES = ['Champions League', 'AFC Champions League', 'Leagues Cup']
    NATIONAL_CUP_NAMES = ['FA Cup', 'Italy Cup', 'Coupe de France', 'DFB-Pokal', 'Copa del Rey', 'Volkswagen Supercup',
                                  'KNVB Beker', 'TFF Süper Kupa', "King's Cup", 'US Open Cup']

    print("Getting player stats for :", name)

    link = player_market_stats[name]['Link']
    scraper = cloudscraper.create_scraper(delay=10, browser={"custom":"ScraperBot/1.0"})
    response = scraper.get(link)
    player_soup = bs.BeautifulSoup(response.content, 'html.parser')
    player_box = player_soup.find_all('div', class_="large-8 columns")
    player_stats_link_a = player_box[0].find_all('a', href=True, class_="content-link")
    stats_link = player_stats_link_a[0]['href'].split('#')[0] + "/plus/1#gesamt"
    player_stats_link = "https://www.transfermarkt.com/" + stats_link

    #Scrape player stats page
    scraper = cloudscraper.create_scraper(delay=10, browser={"custom":"ScraperBot/1.0"})
    response = scraper.get(player_stats_link)
    stats_soup = bs.BeautifulSoup(response.content, 'html.parser')

    #Get stats table headers
    try:
        stats_div = stats_soup.find_all('div', class_="large-12 columns")
        stats_box = stats_div[0].find_all('div', class_="box")[1]
        stats_table = stats_box.find_all('table', class_="items")
        stats_table_headers = stats_table[0].find_all('th')

        stats_table_headers_a =  []
        for header in stats_table_headers : 
            stats_table_headers_a +=header.find_all('a')
        if stats_table_headers_a == []:
            stats_headers = [header.text if header.text != "\xa0" else header.find('span').get('title') for header in stats_table_headers]
        else:
            stats_headers = [header.text if header.text != "\xa0" else header.find('span').get('title') for header in stats_table_headers_a]
        stats_headers.remove('wettbewerb')

        player_stats = {}
    
        for header in stats_headers:
            player_stats[header] = []
        
        #Get stats table rows data
        stats_table_rows = stats_table[0].find_all('tr')
        stats_table_rows_without_header = stats_table_rows[1:]
        stats_rows_data = []
        for row in stats_table_rows_without_header:
            rows_data = row.find_all('td')
            stats_rows_data.append([data.text.strip('\n').replace("\xa0", "").replace("-", "Na").strip(":").strip(" ") if data.text.strip('\n')
                .strip('\n').replace("\xa0", "").replace("-", "Na").strip(":").strip(" ")  != '' else None for data in rows_data])
        
        
        #Fill player_stats dict with data
        for data in stats_rows_data:
            for d in data:
                if d == None:
                    data.remove(d)
        
        for data in stats_rows_data:
            for i in range(len(data)):
                player_stats[stats_headers[i]].append(data[i])

    except:
        player_stats = {}
        print("No stats for: ", name)

    index_total = 0
    index_national = 0
    index_international = 0
    index_nationalcup = 0

    

    for key,value in player_stats.items():

        if player_stats[key] == []:
            continue 
        
        for i in range(len(player_stats[key])):
            if player_stats[key][i] == 'Total':
                index_total = i
            elif player_stats[key][i] in NATIONAL_COMPETITION_NAMES:
                index_national = i
                player_stats["League"] = player_stats[key][i]
            elif player_stats[key][i] in INTERNATIONAL_COMPETITION_NAMES:
                index_international = i
            elif player_stats[key][i] in NATIONAL_CUP_NAMES:
                index_nationalcup = i
        
            if (i+1)/(index_total+1) == 1:
                if key == 'Appearances':
                    player_market_stats[name]['Competition_Total_Appearances'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Goals':
                    player_market_stats[name]['Competition_Total_Goals'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Own goals':
                    player_market_stats[name]['Competition_Total_Own_Goals'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Assists':
                    player_market_stats[name]['Competition_Total_Assists'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Yellow cards':
                    player_market_stats[name]['Competition_Total_Yellow_Cards'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Red cards':
                    player_market_stats[name]['Competition_Total_Red_Cards'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Substitutions on':
                    player_market_stats[name]['Competition_Total_Substitutions_On'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Substitutions off':
                    player_market_stats[name]['Competition_Total_Substitutions_Off'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Clean sheets':
                    player_market_stats[name]['Competition_Total_Clean_Sheets'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Goals conceded':
                    player_market_stats[name]['Competition_Total_Goals_Conceded'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Minutes played':
                    player_market_stats[name]['Competition_Total_Minutes_Played'] = player_stats[key][i] if player_stats[key][i] else None
            elif (i+1)/(index_national+1) == 1:
                if key == 'Appearances':
                    player_market_stats[name]['Competition_National_Appearances'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Goals':
                    player_market_stats[name]['Competition_National_Goals'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Own goals':
                    player_market_stats[name]['Competition_National_Own_Goals'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Assists':
                    player_market_stats[name]['Competition_National_Assists'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Yellow cards':
                    player_market_stats[name]['Competition_National_Yellow_Cards'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Red cards':
                    player_market_stats[name]['Competition_National_Red_Cards'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Substitutions on':
                    player_market_stats[name]['Competition_National_Substitutions_On'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Substitutions off':
                    player_market_stats[name]['Competition_National_Substitutions_Off'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Clean sheets':
                    player_market_stats[name]['Competition_National_Clean_Sheets'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Goals conceded':
                    player_market_stats[name]['Competition_National_Goals_Conceded'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Minutes played':
                    player_market_stats[name]['Competition_National_Minutes_Played'] = player_stats[key][i] if player_stats[key][i] else None
            elif (i+1)/(index_international+1) == 1:
                if key == 'Appearances':
                    player_market_stats[name]['Competition_International_Appearances'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Goals':
                    player_market_stats[name]['Competition_International_Goals'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Own goals':
                    player_market_stats[name]['Competition_International_Own_Goals'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Assists':
                    player_market_stats[name]['Competition_International_Assists'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Yellow cards':
                    player_market_stats[name]['Competition_International_Yellow_Cards'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Red cards':
                    player_market_stats[name]['Competition_International_Red_Cards'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Substitutions on':
                    player_market_stats[name]['Competition_International_Substitutions_On'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Substitutions off':
                    player_market_stats[name]['Competition_International_Substitutions_Off'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Clean sheets':
                    player_market_stats[name]['Competition_International_Clean_Sheets'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Goals conceded':
                    player_market_stats[name]['Competition_International_Goals_Conceded'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Minutes played':
                    player_market_stats[name]['Competition_International_Minutes_Played'] = player_stats[key][i] if player_stats[key][i] else None
            elif (i+1)/(index_nationalcup+1) == 1:
                if key == 'Appearances':
                    player_market_stats[name]['Competition_Nationalcup_Appearances'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Goals':
                    player_market_stats[name]['Competition_Nationalcup_Goals'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Own goals':
                    player_market_stats[name]['Competition_Nationalcup_Own_Goals'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Assists':
                    player_market_stats[name]['Competition_Nationalcup_Assists'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Yellow cards':
                    player_market_stats[name]['Competition_Nationalcup_Yellow_Cards'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Red cards':
                    player_market_stats[name]['Competition_Nationalcup_Red_Cards'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Substitutions on':
                    player_market_stats[name]['Competition_Nationalcup_Substitutions_On'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Substitutions off':
                    player_market_stats[name]['Competition_Nationalcup_Substitutions_Off'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Clean sheets':
                    player_market_stats[name]['Competition_Nationalcup_Clean_Sheets'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Goals conceded':
                    player_market_stats[name]['Competition_Nationalcup_Goals_Conceded'] = player_stats[key][i] if player_stats[key][i] else None
                elif key == 'Minutes played':
                    player_market_stats[name]['Competition_Nationalcup_Minutes_Played'] = player_stats[key][i] if player_stats[key][i] else None

    print("Stats retrieved for :", name)
    return player_market_stats[name]

def get_transfer_market_data_market_value_for_competition(url):

    #Scrape data from web url
    print("Data scraping will begin now")

    players_collected_data = {}

    #Get page content
    scraper = cloudscraper.create_scraper(delay=10, browser={"custom":"ScraperBot/1.0"})
    response = scraper.get(url)#, headers=headers)
    soup = bs.BeautifulSoup(response.content, 'html.parser')
    club_table = soup.find_all('table', class_="items")
    club_list = club_table[0].find_all('tbody')
    club_links_list = club_list[0].find_all('td', class_="hauptlink no-border-links")
    club_links = []

    for clubs in club_links_list:
        club_links += clubs.find_all('a', href=True)

    clubs = {}

    for c_links in club_links:
        club_link = URL_ROOT + c_links['href']
        time.sleep(2)

        response = scraper.get(club_link)
        club_soup = bs.BeautifulSoup(response.content, 'html.parser')
        try:
            players_tbody = club_soup.find_all('table', class_="items")
            players_list = players_tbody[0].find_all('tbody')
        except:
            players_tbod_div = club_soup.find_all('div', class_="large-8 columns")
            players_tbody = players_tbod_div[0].find_all('table', class_="items")
            players_list = players_tbody[0].find_all('tbody')

        
        players_market_values_list = players_list[0].find_all('td', class_="rechts hauptlink")

        players_table = players_list[0].find_all('table', class_="inline-table")
        players_data = []

        for table in players_table:
            players_data += table.find_all('a', href=True)

        players = {}

        for i in range(len(players_data)):
            player_name = players_data[i].text.strip()
            try:
                players_collected_data[player_name] = {'Market value' : players_market_values_list[i].find('a').contents}
            except:
                players_collected_data[player_name] = {'Market value' : None}
            players[player_name] = URL_ROOT + players_data[i]['href']




        for name,p_link in players.items():
            print("Getting player market value stats for: ", name)
            time.sleep(1)
            new_player_data = players_collected_data[name]
            new_player_data['Link'] = p_link
            response = scraper.get(p_link)
            player_soup = bs.BeautifulSoup(response.content, 'html.parser')
            player_box = player_soup.find_all('div', class_="large-8 columns")
            try :
                player_data_div = player_box[0].find_all('div', class_="info-table info-table--right-space")
                player_data_spans = player_data_div[0].find_all('span')
            except IndexError:
                try : 
                    player_data_div = player_box[0].find_all('div', class_="info-table info-table--right-space min-height-audio")
                    player_data_spans = player_data_div[0].find_all('span')
                except IndexError:
                    print('No data found for player:', name)
                    players_collected_data[name] = new_player_data
                    continue

            player_details = ['Name in home country:', 'Date of birth/Age:', 'Place of birth:', 'Height:', 'Foot:', 'Citizenship:', 'Position:', 'Player agent:', 'Current club:', 'Joined:', 'Market value:', 'Contract expires:']
            player_data = {}


            for i in range(len(player_data_spans)):
                data = player_data_spans[i].text.strip()
                #print(data)
                if data in player_details:
                    new_player_data[player_data_spans[i].text.strip()] = player_data_spans[i+1].text.strip().replace('"\xa0"', '')


            players_collected_data[name] = new_player_data
            players_collected_data[name] = get_transfer_market_data_player_stats_for_competition(players_collected_data, name)
            print("Player market value stats retrieved for: ", name)
            
    return players_collected_data

def player_stats_to_csv(players_collected_data, i):
    
    d = {
        'Link' : [],
        'Club' : [],
        'Name' : [],
        'Market value' : [],
        'Date of birth/Age' : [],
        'Place of birth' : [],
        'Citizenship' : [],
        'Height' : [],
        'Foot' : [],
        'Position' : [],
        'Player agent' : [],
        'Joined' : [],
        'Contract expires' : [],
        'Competition_Total_Appearances' : [],
        'Competition_Total_Goals' : [],
        'Competition_Total_Own_Goals' : [],
        'Competition_Total_Assists' : [],
        'Competition_Total_Yellow_Cards' : [],
        'Competition_Total_Red_Cards' : [],
        'Competition_Total_Substitutions_On' : [],
        'Competition_Total_Substitutions_Off' : [],
        'Competition_Total_Clean_Sheets' : [],
        'Competition_Total_Goals_Conceded' : [],
        'Competition_Total_Minutes_Played' : [],
        'Competition_National_Appearances' : [],
        'Competition_National_Goals' : [],
        'Competition_National_Own_Goals' : [],
        'Competition_National_Assists' : [],
        'Competition_National_Yellow_Cards' : [],
        'Competition_National_Red_Cards' : [],
        'Competition_National_Substitutions_On' : [],
        'Competition_National_Substitutions_Off' : [],
        'Competition_National_Goals_Conceded' : [],
        'Competition_National_Clean_Sheets' : [],
        'Competition_National_Minutes_Played' : [],
        'Competition_International_Appearances' : [],
        'Competition_International_Goals' : [],
        'Competition_International_Own_Goals' : [],
        'Competition_International_Assists' : [],
        'Competition_International_Yellow_Cards' : [],
        'Competition_International_Red_Cards' : [],
        'Competition_International_Substitutions_On' : [],
        'Competition_International_Substitutions_Off' : [],
        'Competition_International_Clean_Sheets' : [],
        'Competition_International_Goals_Conceded' : [],
        'Competition_International_Minutes_Played' : [],
        'Competition_Nationalcup_Appearances' : [],
        'Competition_Nationalcup_Goals' : [],
        'Competition_Nationalcup_Own_Goals' : [],
        'Competition_Nationalcup_Assists' : [],
        'Competition_Nationalcup_Yellow_Cards' : [],
        'Competition_Nationalcup_Red_Cards' : [],
        'Competition_Nationalcup_Substitutions_On' : [],
        'Competition_Nationalcup_Substitutions_Off' : [],
        'Competition_Nationalcup_Clean_Sheets' : [],
        'Competition_Nationalcup_Goals_Conceded' : [],
        'Competition_Nationalcup_Minutes_Played' : []
    }


    for k,p in players_collected_data.items():
        try :
            d['Link'].append(p['Link'])
        except:
            d['Link'].append(None)
        try :
            d['Club'].append(p['Current club:'])
        except:
            d['Club'].append(None)
        try:
            d['Name'].append(k)
        except:
            d['Name'].append(None)
        try:
            d['Market value'].append(p['Market value'][0])
        except:
            d['Market value'].append(None)
        try:
            d['Date of birth/Age'].append(p['Date of birth/Age:'])
        except:
            d['Date of birth/Age'].append(None)
        try:
            d['Place of birth'].append(p['Place of birth:'])
        except:
            d['Place of birth'].append(None)
        try:
            d['Citizenship'].append(p['Citizenship:'])
        except:
            d['Citizenship'].append(None)
        try:
            d['Height'].append(p['Height:'])
        except:
            d['Height'].append(None)
        try:
            d['Foot'].append(p['Foot:'])
        except:
            d['Foot'].append(None)
        try:
            d['Position'].append(p['Position:'])
        except:
            d['Position'].append(None)
        try:
            d['Player agent'].append(p['Player agent:'])
        except:
            d['Player agent'].append(None)
        try:
            d['Joined'].append(p['Joined:'])
        except:
            d['Joined'].append(None)
        try:
            d['Contract expires'].append(p['Contract expires:'])
        except:
            d['Contract expires'].append(None)
        try :
            d['Competition_Total_Appearances'].append(p['Competition_Total_Appearances'])
        except:
            d['Competition_Total_Appearances'].append(None)
        try :
            d['Competition_Total_Goals'].append(p['Competition_Total_Goals'])
        except:
            d['Competition_Total_Goals'].append(None)
        try:
            d['Competition_Total_Own_Goals'].append(p['Competition_Total_Own_Goals'])
        except:
            d['Competition_Total_Own_Goals'].append(None)
        try:
            d['Competition_Total_Assists'].append(p['Competition_Total_Assists'])
        except:
            d['Competition_Total_Assists'].append(None)
        try:
            d['Competition_Total_Yellow_Cards'].append(p['Competition_Total_Yellow_Cards'])
        except:
            d['Competition_Total_Yellow_Cards'].append(None)
        try:
            d['Competition_Total_Red_Cards'].append(p['Competition_Total_Red_Cards'])
        except:
            d['Competition_Total_Red_Cards'].append(None)
        try:
            d['Competition_Total_Substitutions_On'].append(p['Competition_Total_Substitutions_On'])
        except:
            d['Competition_Total_Substitutions_On'].append(None)
        try:
            d['Competition_Total_Substitutions_Off'].append(p['Competition_Total_Substitutions_Off'])
        except:
            d['Competition_Total_Substitutions_Off'].append(None)
        try:
            d['Competition_Total_Clean_Sheets'].append(p['Competition_Total_Clean_Sheets'])
        except:
            d['Competition_Total_Clean_Sheets'].append(None)
        try:
            d['Competition_Total_Goals_Conceded'].append(p['Competition_Total_Goals_Conceded'])
        except:
            d['Competition_Total_Goals_Conceded'].append(None)
        try:
            d['Competition_Total_Minutes_Played'].append(p['Competition_Total_Minutes_Played'])
        except:
            d['Competition_Total_Minutes_Played'].append(None)
        try :
            d['Competition_National_Appearances'].append(p['Competition_National_Appearances'])
        except:
            d['Competition_National_Appearances'].append(None)
        try :
            d['Competition_National_Goals'].append(p['Competition_National_Goals'])
        except:
            d['Competition_National_Goals'].append(None)
        try:
            d['Competition_National_Own_Goals'].append(p['Competition_National_Own_Goals'])
        except:
            d['Competition_National_Own_Goals'].append(None)
        try:
            d['Competition_National_Assists'].append(p['Competition_National_Assists'])
        except:
            d['Competition_National_Assists'].append(None)
        try:
            d['Competition_National_Yellow_Cards'].append(p['Competition_National_Yellow_Cards'])
        except:
            d['Competition_National_Yellow_Cards'].append(None)
        try:
            d['Competition_National_Red_Cards'].append(p['Competition_National_Red_Cards'])
        except:
            d['Competition_National_Red_Cards'].append(None)
        try:
            d['Competition_National_Substitutions_On'].append(p['Competition_National_Substitutions_On'])
        except:
            d['Competition_National_Substitutions_On'].append(None)
        try:
            d['Competition_National_Substitutions_Off'].append(p['Competition_National_Substitutions_Off'])
        except:
            d['Competition_National_Substitutions_Off'].append(None)
        try:
            d['Competition_National_Clean_Sheets'].append(p['Competition_National_Clean_Sheets'])
        except:
            d['Competition_National_Clean_Sheets'].append(None)
        try:
            d['Competition_National_Goals_Conceded'].append(p['Competition_National_Goals_Conceded'])
        except:
            d['Competition_National_Goals_Conceded'].append(None)
        try:
            d['Competition_National_Minutes_Played'].append(p['Competition_National_Minutes_Played'])
        except:
            d['Competition_National_Minutes_Played'].append(None)
        try :
            d['Competition_International_Appearances'].append(p['Competition_International_Appearances'])
        except:
            d['Competition_International_Appearances'].append(None)
        try :
            d['Competition_International_Goals'].append(p['Competition_International_Goals'])
        except:
            d['Competition_International_Goals'].append(None)
        try:
            d['Competition_International_Own_Goals'].append(p['Competition_International_Own_Goals'])
        except:
            d['Competition_International_Own_Goals'].append(None)
        try:
            d['Competition_International_Assists'].append(p['Competition_International_Assists'])
        except:
            d['Competition_International_Assists'].append(None)
        try:
            d['Competition_International_Yellow_Cards'].append(p['Competition_International_Yellow_Cards'])
        except:
            d['Competition_International_Yellow_Cards'].append(None)
        try:
            d['Competition_International_Red_Cards'].append(p['Competition_International_Red_Cards'])
        except:
            d['Competition_International_Red_Cards'].append(None)
        try:
            d['Competition_International_Substitutions_On'].append(p['Competition_International_Substitutions_On'])
        except:
            d['Competition_International_Substitutions_On'].append(None)
        try:
            d['Competition_International_Substitutions_Off'].append(p['Competition_International_Substitutions_Off'])
        except:
            d['Competition_International_Substitutions_Off'].append(None)
        try:
            d['Competition_International_Clean_Sheets'].append(p['Competition_International_Clean_Sheets'])
        except:
            d['Competition_International_Clean_Sheets'].append(None)
        try:
            d['Competition_International_Goals_Conceded'].append(p['Competition_International_Goals_Conceded'])
        except:
            d['Competition_International_Goals_Conceded'].append(None)
        try:
            d['Competition_International_Minutes_Played'].append(p['Competition_International_Minutes_Played'])
        except:
            d['Competition_International_Minutes_Played'].append(None)
        try :
            d['Competition_Nationalcup_Appearances'].append(p['Competition_Nationalcup_Appearances'])
        except:
            d['Competition_Nationalcup_Appearances'].append(None)
        try :
            d['Competition_Nationalcup_Goals'].append(p['Competition_Nationalcup_Goals'])
        except:
            d['Competition_Nationalcup_Goals'].append(None)
        try:
            d['Competition_Nationalcup_Own_Goals'].append(p['Competition_Nationalcup_Own_Goals'])
        except:
            d['Competition_Nationalcup_Own_Goals'].append(None)
        try:
            d['Competition_Nationalcup_Assists'].append(p['Competition_Nationalcup_Assists'])
        except:
            d['Competition_Nationalcup_Assists'].append(None)
        try:
            d['Competition_Nationalcup_Yellow_Cards'].append(p['Competition_Nationalcup_Yellow_Cards'])
        except:
            d['Competition_Nationalcup_Yellow_Cards'].append(None)
        try:
            d['Competition_Nationalcup_Red_Cards'].append(p['Competition_Nationalcup_Red_Cards'])
        except:
            d['Competition_Nationalcup_Red_Cards'].append(None)
        try:
            d['Competition_Nationalcup_Substitutions_On'].append(p['Competition_Nationalcup_Substitutions_On'])
        except:
            d['Competition_Nationalcup_Substitutions_On'].append(None)
        try:
            d['Competition_Nationalcup_Substitutions_Off'].append(p['Competition_Nationalcup_Substitutions_Off'])
        except:
            d['Competition_Nationalcup_Substitutions_Off'].append(None)
        try:
            d['Competition_Nationalcup_Clean_Sheets'].append(p['Competition_Nationalcup_Clean_Sheets'])
        except:
            d['Competition_Nationalcup_Clean_Sheets'].append(None)
        try:
            d['Competition_Nationalcup_Goals_Conceded'].append(p['Competition_Nationalcup_Goals_Conceded'])
        except:
            d['Competition_Nationalcup_Goals_Conceded'].append(None)
        try:
            d['Competition_Nationalcup_Minutes_Played'].append(p['Competition_Nationalcup_Minutes_Played'])
        except:
            d['Competition_Nationalcup_Minutes_Played'].append(None)

    df = pd.DataFrame(d)
    df.to_csv('data/scraped-football-player-market-values/Players_transfer_market_data_complete' + str(i) + '.csv', encoding='utf-8', index=False)
    print(f"The data has been saved in 'data/scraped-football-player-market-values/Players_transfer_market_data{str(i)}.csv")


for I in range(len(URL)):
    player_stats_to_csv(get_transfer_market_data_market_value_for_competition(URL[i]), i)
    i+=1

files = glob.glob("data/scraped-football-player-market-values/*.csv")

files_to_concat = [f for f in files if f not in [
    'data/scraped-football-player-market-values\\Players_transfer_market_data_complete.csv', 
    'data/scraped-football-player-market-values\\Players_transfer_market_data_complete_prepared.csv'
    ]]

df_list = [pd.read_csv(f) for f in files_to_concat]

df = pd.concat(df_list)

df.to_csv("data/scraped-football-player-market-values/Players_transfer_market_data_complete.csv", index=False)

for f in files_to_concat:
    os.remove(f)


#get_transfer_market_data_player_stats_for_competition('Players_transfer_market_data.csv')