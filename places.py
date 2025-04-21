import sqlite3
from dataclasses import dataclass
from typing import List, Optional
import googlemaps
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Data Classes for Type Safety
@dataclass
class Specialist:
    id: int
    name: str
    description: str

@dataclass
class Disease:
    id: int
    name: str
    specialist_id: int
    symptoms: str

@dataclass
class Hospital:
    name: str
    address: str
    rating: float
    specialist_available: bool
    distance: float
    place_id: str

class MedicalPlacesSystem:
    def __init__(self):
        """Initialize the system with database and Google Maps client"""
        self.db_path = "medical_db.sqlite"
        print(f"Initializing MedicalPlacesSystem with database: {self.db_path}")
        
        # Load and verify Google Maps API key
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")
        print(f"Google Maps API key loaded: {api_key[:10]}...")
        
        self.gmaps = googlemaps.Client(key=api_key)
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with necessary tables"""
        print("Initializing database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create Specialists table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS specialists (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT
        )
        ''')

        # Create Diseases table with specialist mapping
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS diseases (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            specialist_id INTEGER,
            symptoms TEXT,
            FOREIGN KEY (specialist_id) REFERENCES specialists (id)
        )
        ''')

        # Insert some initial data
        self._insert_initial_data(cursor)
        
        # Verify data insertion
        cursor.execute("SELECT COUNT(*) FROM specialists")
        specialist_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM diseases")
        disease_count = cursor.fetchone()[0]
        print(f"Database initialized with {specialist_count} specialists and {disease_count} diseases")
        
        conn.commit()
        conn.close()

    def _insert_initial_data(self, cursor):
        """Insert initial specialist and disease data"""
        # Insert specialists if they don't exist
        specialists = [
    (1, "Cardiologist", "Heart and blood vessel specialist"),
    (2, "Dermatologist", "Skin specialist"),
    (3, "Neurologist", "Brain and nervous system specialist"),
    (4, "Infectious Disease Specialist", "Specialist in treating infectious diseases including rabies"),
    (5, "Endocrinologist", "Hormone and metabolic disorder specialist"),
    (6, "Pulmonologist", "Lung and respiratory specialist"),
    (7, "Internal Medicine Specialist", "General physician for internal organs"),
    (8, "Rheumatologist", "Arthritis and autoimmune disease specialist"),
    (9, "Pulmonologist", "Lung and respiratory system specialist"),
    (10, "Hepatologist", "Liver specialist"),
    (11, "Psychiatrist", "Mental health specialist"),
    (12, "Ophthalmologist", "Eye specialist"),
    (13, "Virologist", "Virus and infectious disease specialist"),
    (14, "Gastroenterologist", "Digestive system specialist"),
    (15, "Toxicologist", "Expert in poisoning and food contamination"),
    (16, "Orthopedic Specialist", "Bone and musculoskeletal specialist"),
    (17, "Nephrologist", "Kidney specialist"),
    (18, "Allergist/Immunologist", "Specialist in allergies and immune system disorders"),
    (19, "Oncologist", "Cancer specialist"),
    (20, "Otolaryngologist (ENT)", "Ear, nose, and throat specialist"),
    (21, "Urologist", "Urinary tract and male reproductive system specialist"),
    (22, "Gynecologist", "Female reproductive system specialist"),
    (23, "Pediatrician", "Child health specialist"),
    (24, "Hematologist", "Blood disease specialist"),
    (25, "Dentist", "Oral health and teeth specialist"),
    (26, "Genetic Counselor", "Specialist in genetic disorders and family history analysis"),
    (27, "Nutritionist", "Expert in dietary planning and nutritional guidance"),
    (28, "Physiotherapist", "Specialist in physical rehabilitation and exercise therapy"),
    (29, "Neurosurgeon", "Specialist in brain and spinal surgeries"),
    (30, "Radiologist", "Expert in interpreting medical images such as X-rays, CT scans, and MRIs"),
    (31, "Pathologist", "Specialist in diagnosing diseases based on laboratory analysis of bodily fluids and tissues")
    
    
]

        cursor.executemany('''
        INSERT OR IGNORE INTO specialists (id, name, description)
        VALUES (?, ?, ?)
        ''', specialists)

        # Insert diseases if they don't exist
        diseases = [
    (1, "Heart Disease", 1, "chest pain,shortness of breath"),
    (2, "Eczema", 2, "itchy skin,redness"),
    (3, "Migraine", 3, "headache,sensitivity to light"),
    (4, "Rabies", 4, "fever,headache,anxiety,confusion,hyperactivity,difficulty swallowing"),
    (5, "Diabetes", 5, "increased thirst,frequent urination,extreme hunger,unexplained weight loss"),
    (6, "Asthma", 6, "shortness of breath,wheezing,coughing,chest tightness"),
    (7, "Pneumonia", 7, "fever,chills,cough with phlegm,shortness of breath"),
    (8, "Hypertension", 1, "high blood pressure,headache,dizziness,blurred vision"),
    (9, "Arthritis", 8, "joint pain,stiffness,swelling,decreased range of motion"),
    (10, "Epilepsy", 3, "seizures,uncontrollable jerking,loss of awareness,confusion"),
    (11, "Tuberculosis", 9, "persistent cough,weight loss,night sweats,fever"),
    (12, "Hepatitis", 10, "jaundice,fatigue,nausea,abdominal pain,dark urine"),
    (13, "Depression", 11, "persistent sadness,loss of interest,sleep disturbances,fatigue"),
    (14, "Glaucoma", 12, "eye pain,blurred vision,headache,nausea"),
    (15, "COVID-19", 13, "fever,cough,shortness of breath,fatigue,loss of taste or smell"),
    (16, "Stroke", 3, "sudden numbness,weakness,confusion,trouble speaking"),
    (17, "Acid Reflux", 14, "heartburn,regurgitation,burping,throat irritation"),
    (18, "Food Poisoning", 15, "nausea,vomiting,diarrhea,stomach cramps,fever"),
    (19, "Osteoporosis", 16, "bone fractures,back pain,decreased height,stooped posture"),
    (20, "Parkinson's Disease", 3, "tremors,stiffness,slow movement,loss of balance"),
    (21, "Chronic Kidney Disease", 17, "fatigue,swelling in ankles,urine changes,shortness of breath"),
    (22, "Allergic Rhinitis", 18, "sneezing,runny nose,itchy eyes,nasal congestion"),
    (23, "Leukemia", 19, "frequent infections,fatigue,weight loss,easy bruising"),
    (24, "Tonsillitis", 20, "sore throat,fever,difficulty swallowing,swollen tonsils"),
    (25, "Urinary Tract Infection", 21, "burning sensation during urination,frequent urge to urinate,cloudy urine"),
    (26, "Polycystic Ovary Syndrome (PCOS)", 22, "irregular periods,acne,weight gain,excess hair growth"),
    (27, "Autism Spectrum Disorder", 23, "difficulty with communication,repetitive behaviors,delayed development"),
    (28, "Anemia", 24, "fatigue,pale skin,shortness of breath,dizziness"),
    (29, "Tooth Decay", 25, "toothache,sensitivity,visible holes in teeth"),
    (30, "Otitis Media", 20, "ear pain,hearing difficulty,fever,fluid drainage from ear"),
    (31, "Dengue Fever", 4, "high fever,severe headache,joint and muscle pain,rash,bleeding gums"),
    (32, "Malaria", 4, "cyclical fever,chills,headache,muscle pain,anemia"),
    (33, "Lung Cancer", 19, "persistent cough,chest pain,weight loss,shortness of breath,hemoptysis"),
    (34, "Acute Gastroenteritis", 14, "watery diarrhea,vomiting,abdominal cramps,fever"),
    (35, "Obesity", 5, "excessive weight,fatigue,joint pain,sleep disturbances"),
    (36, "COPD", 6, "chronic cough,shortness of breath,frequent respiratory infections,wheezing"),
    (37, "Kidney Stones", 17, "severe back pain,blood in urine,nausea,vomiting"),
    (38, "Appendicitis", 7, "abdominal pain,fever,nausea,loss of appetite"),
    (39, "Hyperthyroidism", 5, "rapid heartbeat,weight loss,anxiety,tremors,heat intolerance"),
    (40, "Hypothyroidism", 5, "fatigue,weight gain,cold intolerance,dry skin,constipation")
]


        cursor.executemany('''
        INSERT OR IGNORE INTO diseases (id, name, specialist_id, symptoms)
        VALUES (?, ?, ?, ?)
        ''', diseases)

    def get_specialist_for_disease(self, disease_name: str) -> Optional[Specialist]:
        """Find the appropriate specialist for a given disease or symptoms"""
        print(f"Searching for specialist for disease/symptoms: {disease_name}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First try exact disease name match
        cursor.execute('''
        SELECT s.* FROM specialists s
        JOIN diseases d ON d.specialist_id = s.id
        WHERE LOWER(d.name) LIKE LOWER(?)
        ''', (f"%{disease_name}%",))
        
        result = cursor.fetchone()
        
        # If no result, try matching symptoms
        if not result and ',' in disease_name:
            symptoms = [s.strip().lower() for s in disease_name.split(',')]
            symptom_placeholders = ','.join(['?' for _ in symptoms])
            
            cursor.execute(f'''
            SELECT s.*, COUNT(DISTINCT d.id) as symptom_matches
            FROM specialists s
            JOIN diseases d ON d.specialist_id = s.id
            WHERE EXISTS (
                SELECT 1 
                FROM diseases d2 
                WHERE d2.specialist_id = s.id 
                AND (
                    {' OR '.join([f"LOWER(d2.symptoms) LIKE ?" for _ in symptoms])}
                )
            )
            GROUP BY s.id
            ORDER BY symptom_matches DESC
            LIMIT 1
            ''', [f"%{symptom}%" for symptom in symptoms])
            
            result = cursor.fetchone()
        
        conn.close()
        
        if result:
            print(f"Found specialist: {result[1]}")
            return Specialist(id=result[0], name=result[1], description=result[2])
        print("No specialist found")
        return None

    def find_nearby_hospitals(self, location: str, specialist_type: str) -> List[Hospital]:
        """Find hospitals near a given location that have the required specialist"""
        try:
            print(f"Searching for hospitals in {location} with {specialist_type}")
            # First, geocode the location
            geocode_result = self.gmaps.geocode(location)
            if not geocode_result:
                print("Geocoding failed")
                return []

            lat_lng = geocode_result[0]['geometry']['location']
            print(f"Location coordinates: {lat_lng}")

            # For rabies, search for emergency hospitals or infectious disease centers
            if specialist_type.lower() == "infectious disease specialist":
                search_keywords = ["hospital emergency", "infectious disease center", "emergency room"]
            else:
                search_keywords = [f"hospital {specialist_type}"]

            print(f"Using search keywords: {search_keywords}")
            all_results = []
            for keyword in search_keywords:
                places_result = self.gmaps.places_nearby(
                    location=lat_lng,
                    keyword=keyword,
                    radius=5000,  # 5km radius
                    type='hospital'
                )
                if places_result.get('results'):
                    all_results.extend(places_result['results'])

            print(f"Found {len(all_results)} total results")
            # Remove duplicates based on place_id
            seen_place_ids = set()
            unique_results = []
            for result in all_results:
                if result['place_id'] not in seen_place_ids:
                    seen_place_ids.add(result['place_id'])
                    unique_results.append(result)

            print(f"After removing duplicates: {len(unique_results)} unique results")
            hospitals = []
            for place in unique_results:
                hospital = Hospital(
                    name=place['name'],
                    address=place.get('vicinity', 'Address not available'),
                    rating=place.get('rating', 0.0),
                    specialist_available=True,
                    distance=0.0,
                    place_id=place['place_id']
                )
                hospitals.append(hospital)

            # Sort by rating
            hospitals.sort(key=lambda x: x.rating, reverse=True)
            print(f"Returning top {min(5, len(hospitals))} hospitals")
            return hospitals[:5]  # Return top 5 hospitals

        except Exception as e:
            print(f"Error finding hospitals: {str(e)}")
            return []

    def get_recommendations(self, disease_name: str, location: str) -> dict:
        """Get hospital recommendations based on disease and location"""
        print(f"\nGetting recommendations for {disease_name} in {location}")
        # Find specialist for the disease
        specialist = self.get_specialist_for_disease(disease_name)
        if not specialist:
            print(f"No specialist found for {disease_name}")
            return {
                "success": False,
                "message": f"I couldn't find any specialists for {disease_name}."
            }

        # Find nearby hospitals with the specialist
        hospitals = self.find_nearby_hospitals(location, specialist.name)
        if not hospitals:
            print(f"No hospitals found for {specialist.name} in {location}")
            return {
                "success": False,
                "message": f"I couldn't find any hospitals with {specialist.name} near {location}."
            }

        print(f"Found {len(hospitals)} hospitals")
        
        # Use HTML formatting to ensure proper spacing
        initial_response = f"<div class='medical-help-result'>"
        initial_response += f"<h3>Medical Help for {disease_name}</h3>"
        initial_response += f"<p>Based on your symptoms, you should consult a:</p>"
        initial_response += f"<p class='specialist-name'><strong>{specialist.name}</strong></p>"
        initial_response += f"<h4>Recommended Hospitals in {location}:</h4>"
        initial_response += f"<div class='hospital-list'>"

        for i, hospital in enumerate(hospitals, 1):
            initial_response += f"<div class='hospital-item'>"
            initial_response += f"<p class='hospital-number'>{i}. <strong>{hospital.name}</strong></p>"
            initial_response += f"<p class='hospital-address'>📍 {hospital.address}</p>"
            if hospital.rating > 0:
                # Create stars based on the integer part of the rating
                stars = "⭐" * int(hospital.rating)
                initial_response += f"<p class='hospital-rating'>Rating: {stars} ({hospital.rating}/5)</p>"
            else:
                initial_response += "<p class='hospital-rating'>Rating: Not available</p>"
            initial_response += f"</div>"
        
        initial_response += f"</div></div>"

        # Send to Together API for better formatting
        try:
            import requests
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
            
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Please format this medical recommendation to be concise, clear, and user-friendly. 
            Keep the emojis and stars, maintain the spacing, and ensure it's easy to read:

            {initial_response}"""
            
            data = {
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                "messages": [
                    {"role": "system", "content": "You are a medical information formatter. Format the given medical recommendations to be concise, clear, and user-friendly. Preserve the emojis and formatting."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                formatted_response = response.json()['choices'][0]['message']['content']
                return {
                    "success": True,
                    "response": formatted_response
                }
            else:
                print(f"Together API error: {response.text}")
                return {
                    "success": True,
                    "response": initial_response  # Fallback to original response
                }
                
        except Exception as e:
            print(f"Error formatting with Together API: {str(e)}")
            return {
                "success": True,
                "response": initial_response  # Fallback to original response
            }

# Usage Example
if __name__ == "__main__":
    # Initialize the system
    medical_system = MedicalPlacesSystem()
    
    # Example usage
    result = medical_system.get_recommendations("Heart Disease", "New Delhi, India")
    
    if result["success"]:
        print(result["response"])
    else:
        print(f"Error: {result['message']}")
