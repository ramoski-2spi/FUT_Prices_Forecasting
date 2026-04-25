import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time

#players for automation
PLAYERS = {"1154":"bradley-barcola",
           #"21588":"rayan-cherki",
           #"23955":"jonathan-david",
           #"18729":"pirlo"
           }

def get_price_history(player_id, slug):
    """
    Scrapes FUTBIN price history for a single player.
    Returns a DataFrame with: player_id, player_name, date, price.
    """

    url = f"https://www.futbin.com/26/player/{player_id}/{slug}"
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/123.0.0.0 Safari/537.36"),
                              "Accept-Language": "en-US,en;q=0.9",
                              "Referer": "https://www.google.com/",}

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch page for {slug} (status {response.status_code})")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # FUTBIN stores price history in a hidden attribute called data-pc-data
    div = soup.find(attrs={"data-pc-data": True})

    if div is None:
        print(f"No price history found for {slug}")
        return None

    raw_json = div.get("data-pc-data")

    try:
        price_array = json.loads(raw_json)
    except:
        print(f"Could not parse price data for {slug}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(price_array, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    df["player_id"] = player_id
    df["player_name"] = slug

    return df[["player_id", "player_name", "date", "price"]]


def scrape_players(player_dict):
    """
    Scrapes multiple players and saves the combined CSV.
    """

    all_data = []

    for pid, slug in player_dict.items():
        print(f"Scraping {slug}...")
        df = get_price_history(pid, slug)

        if df is not None:
            all_data.append(df)

        time.sleep(4)  # avoid rate limiting

    if not all_data:
        print("No data scraped.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("data/raw/fut_prices_raw.csv", index=False)
    print("Saved to data/raw/fut_prices_raw.csv")


if __name__ == "__main__":
    #players for the analysis dataset
    players = {"254": "lionel-messi",
           "724": "cristiano-ronaldo-dos-santos-aveiro",
           "41": "florian-wirtz",
           "240": "antonio-rudiger",
           "1009": "alphonso-davies",
           "245": "ruben-santos-gato-alves-dias",
           "86": "joshua-kimmich",
           "952": "marcos-llorente-moreno",
           "17472": "corentin-tolisso",
           "17306": "federico-dimarco",
           "894": "eduardo-camavinga",
           "224": "bruno-miguel-borges-fernandes",
           "68": "vitor-machado-ferreira",
           "56": "federico-valverde",
           "820": "marc-andre-ter-stegen",
           "47": "vinicius-jose-de-oliveira-junior",
           "73": "jude-bellingham",
           "20744": "kylian-mbappe",
           "18918": "erling-haaland",
           "17377": "ronald-araujo",
           "20344": "joao-felix-sequeira",
           "20708": "theo-hernandez",
           "17625": "gabriel-fernando-de-jesus",
           "17442": "daniel-parejo-munoz",
           "22772": "rodrigo-hernandez-cascante",
           "20538": "bernardo-mota-carvalho-e-silva",
           "20705": "jan-oblak",
           "20031": "heung-min-son",
           "21296": "bukayo-saka",
           "21807": "mohamed-salah",
           "795": "antoine-griezmann",
           "23819": "willian-pacho",
           "21495": "pedro-gonzalez-lopez",
           "21615": "paul-pogba",
           "20620": "nuno-albertino-varela-tavares",
           "21762": "nuno-alexandre-tavares-mendes",
           "18693": "zidane",
           "18807": "crespo",
           "21743": "rooney",
           "23859": "toni-kroos",
           "18825": "okocha",
           "21557": "stam",
           "18874": "capdevila-mendez",
           "21647": "declan-rice",
           "20797": "michael-olise",
           "20417": "cody-gakpo",
           "20280": "jordi-alba-ramos",
           "917": "brahim-diaz",
           "17441": "pau-cubarsi-paredes"}

    scrape_players(players)
