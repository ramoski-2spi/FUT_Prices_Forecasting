import requests
import pandas as pd
import time

#Players to test automation pipeline
PLAYERS = {"100914866": "rayan-cherki",
           "50575278": "jonathan-david",
           "100940939": "lamine-yamal"}

def get_price_history(player_id, slug):
    url = f"https://www.fut.gg/api/fut/player-prices/26/{player_id}/"
    
    headers = headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                                        "Chrome/123.0.0.0 Safari/537.36"),
                                        "Accept": "application/json, text/plain, */*",
                                        "Accept-Language": "en-US,en;q=0.9",
                                        "Referer": f"https://www.fut.gg/players/{player_id}-{slug}/",
                                        "Origin": "https://www.fut.gg",
                                        "Connection": "keep-alive",}

    #{"User-Agent": "Mozilla/5.0",
               #"Accept": "application/json"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch for {slug} (status {response.status_code})")
        return None
    
    data = response.json()
    
    # Extract history
    history = data["data"]["history"]
    
    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["player_id"] = player_id
    df["player_name"] = slug   

    return df[["player_id", "player_name", "date", "price"]]

def scrape_players(player_dict):
    all_data = []

    for pid, slug in player_dict.items():
        print(f"Scraping {slug}...")
        df = get_price_history(pid, slug)

        if df is not None:
            all_data.append(df)

        time.sleep(2.5)

    if not all_data:
        print("No data scraped.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("data/raw/fut_prices_raw.csv", index=False)
    print("Saved to data/raw/fut_prices_raw.csv")

if __name__ == "__main__":
    # full FUT.GG ID list
    players = {"158023": "lionel-messi",
           "20801": "cristiano-ronaldo-dos-santos-aveiro",
           "256630": "florian-wirtz",
           "205452": "antonio-rudiger",
           "234396": "alphonso-davies",
           "239818": "ruben-santos-gato-alves-dias",
           "212622": "joshua-kimmich",
           "226161": "marcos-llorente-moreno",
           "219683": "corentin-tolisso",
           "226268": "federico-dimarco",
           "248243": "eduardo-camavinga",
           "212198": "bruno-miguel-borges-fernandes",
           "255253": "vitor-machado-ferreira",
           "239053": "federico-valverde",
           "192448": "marc-andre-ter-stegen",
           "238794": "vinicius-jose-de-oliveira-junior",
           "252371": "jude-bellingham",
           "100895043": "kylian-mbappe",
           "50570733": "erling-haaland",
           "253163": "ronald-araujo",
           "84128524": "joao-felix-sequeira",
           "50564304": "theo-hernandez",
           "230666": "gabriel-fernando-de-jesus",
           "189513": "daniel-parejo-munoz",
           "67340730": "rodrigo-hernandez-cascante",
           "50550315": "bernardo-mota-carvalho-e-silva",
           "50532037": "jan-oblak",
           "50531752": "heung-min-son",
           "50578317": "bukayo-saka",
           "84095411": "mohamed-salah",
           "194765": "antoine-griezmann",
           "84142276": "willian-pacho",
           "50583502": "pedro-gonzalez-lopez",
           "84081944": "paul-pogba",
           "50583178": "nuno-albertino-varela-tavares",
           "50583793": "nuno-alexandre-tavares-mendes",
           "1397": "zidane",
           "7512": "crespo",
           "50385698": "rooney",
           "67291385": "toni-kroos",
           "570": "okocha",
           "50337388": "stam",
           "25924": "capdevila-mendez",
           "84120458": "declan-rice",
           "84133907": "michael-olise",
           "67351380": "cody-gakpo",
           "50520980": "jordi-alba-ramos",
           "231410": "brahim-diaz",
           "278046": "pau-cubarsi-paredes"}

    scrape_players(players)