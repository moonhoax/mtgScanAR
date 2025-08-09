import requests

def get_oracle_cards_url():
    bulk_data_api = "https://api.scryfall.com/bulk-data"
    response = requests.get(bulk_data_api)
    response.raise_for_status()
    bulk_data = response.json()
    oracle_cards_entry = next((item for item in bulk_data['data'] if item['type'] == 'oracle_cards'), None)
    if oracle_cards_entry:
        return oracle_cards_entry['download_uri']
    else:
        raise Exception("Oracle cards bulk data not found.")

def download_file(url, filename):
    print(f"Downloading {filename} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

if __name__ == "__main__":
    url = get_oracle_cards_url()
    download_file(url, "oracle-cards.json")
