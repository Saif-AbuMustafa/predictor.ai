import http.client
import json
import csv

def fetch_historical_data(crypto_id='bitcoin', vs_currency='usd', days='max'):
    conn = http.client.HTTPSConnection("api.coingecko.com")
    endpoint = f"/api/v3/coins/{crypto_id}/market_chart?vs_currency={vs_currency}&days={days}"
    headers = {
        'User-Agent': 'crypto_predictor_bot/1.0'
    }

    # Send the GET request
    conn.request("GET", endpoint, headers=headers)
    response = conn.getresponse()

    if response.status != 200:
        raise Exception(f"Failed to fetch data: {response.status}")
    
    # Load JSON response
    data = json.loads(response.read().decode())
    
    # Assuming 'prices' in JSON data is what we need
    prices = data.get("prices", [])

    # Save to CSV file
    with open("data/crypto_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "price"])
        writer.writerows(prices)
    
    print("Data successfully saved to data/crypto_data.csv")

# Run the function
fetch_historical_data(crypto_id='bitcoin', vs_currency='usd', days='365')
