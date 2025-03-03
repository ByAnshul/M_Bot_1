import requests

# Your OpenFDA API Key
API_KEY = "ZWZ0PGcvbq5gX6bm3FkXnYD6GxMjZ9vlH5S2PoCB"

# Base URL for OpenFDA Drug Event API
BASE_URL = "https://api.fda.gov/drug/event.json"

# Search Query (Modify as Needed)
search_query = "anemia"  # Example: Searching for aspirin-related events

# Full API Request URL
url = f"{BASE_URL}?api_key={API_KEY}&search=patient.drug.medicinalproduct:{search_query}&limit=5"

# Make the Request
response = requests.get(url)

# Check if Request was Successful
if response.status_code == 200:
    data = response.json()
    
    # Extract and Display Results
    for i, result in enumerate(data.get("results", []), start=1):
        print(f"Event {i}:")
        print("  Seriousness:", result.get("serious", "N/A"))
        print("  Report Date:", result.get("receiptdate", "N/A"))
        print("  Reactions:", ", ".join(result.get("patient", {}).get("reaction", [{}])[0].get("reactionmeddrapt", "N/A")))
        print("-" * 50)
else:
    print("Error:", response.status_code, response.text)
