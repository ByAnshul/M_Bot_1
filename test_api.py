import os
from dotenv import load_dotenv
import googlemaps

def test_google_maps_api():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("Error: GOOGLE_MAPS_API_KEY not found in environment variables")
        return
    
    print(f"API Key loaded: {api_key[:10]}...")
    
    try:
        # Initialize Google Maps client
        gmaps = googlemaps.Client(key=api_key)
        
        # Test geocoding
        print("\nTesting Geocoding API...")
        geocode_result = gmaps.geocode("Delhi, India")
        if geocode_result:
            print(f"Geocoding successful for Delhi:")
            print(f"Latitude: {geocode_result[0]['geometry']['location']['lat']}")
            print(f"Longitude: {geocode_result[0]['geometry']['location']['lng']}")
        else:
            print("Geocoding failed")
        
        # Test Places API
        print("\nTesting Places API...")
        places_result = gmaps.places_nearby(
            location=(28.6139, 77.2090),  # Delhi coordinates
            keyword="hospital",
            radius=5000,
            type='hospital'
        )
        
        if places_result.get('results'):
            print(f"Found {len(places_result['results'])} hospitals")
            print("\nTop 3 hospitals:")
            for i, place in enumerate(places_result['results'][:3], 1):
                print(f"\n{i}. {place['name']}")
                print(f"   Address: {place.get('vicinity', 'N/A')}")
                print(f"   Rating: {place.get('rating', 'N/A')}")
        else:
            print("No hospitals found")
            
    except Exception as e:
        print(f"Error testing API: {str(e)}")

if __name__ == "__main__":
    test_google_maps_api() 