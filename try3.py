
import googlemaps
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_google_maps_api():
    try:
        # Initialize Google Maps client
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            print("❌ Error: GOOGLE_MAPS_API_KEY not found in .env file")
            return
            
        print(f"✅ API Key loaded: {api_key[:10]}...")
        
        gmaps = googlemaps.Client(key=api_key)
        
        # Test 1: Geocoding
        print("\n🔍 Testing Geocoding API...")
        location = "Berlin, Germany"
        geocode_result = gmaps.geocode(location)
        
        if geocode_result:
            print(f"✅ Geocoding successful for {location}")
            print(f"Latitude: {geocode_result[0]['geometry']['location']['lat']}")
            print(f"Longitude: {geocode_result[0]['geometry']['location']['lng']}")
        else:
            print("❌ Geocoding failed")
            
        # Test 2: Places API
        print("\n🔍 Testing Places API...")
        if geocode_result:
            lat_lng = geocode_result[0]['geometry']['location']
            places_result = gmaps.places_nearby(
                location=lat_lng,
                keyword="hospital",
                radius=5000,
                type='hospital'
            )
            
            if places_result.get('results'):
                print(f"✅ Found {len(places_result['results'])} hospitals")
                print("\nTop 3 hospitals:")
                for i, hospital in enumerate(places_result['results'][:3], 1):
                    print(f"\n{i}. {hospital['name']}")
                    print(f"   Address: {hospital.get('vicinity', 'N/A')}")
                    print(f"   Rating: {hospital.get('rating', 'N/A')}")
            else:
                print("❌ No hospitals found")
        else:
            print("❌ Places API test skipped due to geocoding failure")
            
    except Exception as e:
        print(f"❌ Error testing Google Maps API: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Google Maps API Test...")
    test_google_maps_api()
