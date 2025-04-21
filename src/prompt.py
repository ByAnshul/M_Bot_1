from flask import session
from src.database import get_db_connection

def get_user_health_info():
    """Retrieve user's health information from the database"""
    if 'user_id' not in session:
        return None, None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symptoms, diseases FROM users WHERE id = ?", (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return user['symptoms'], user['diseases']
    return None, None

def format_health_context(symptoms, diseases):
    """Format health information for the prompt"""
    context = []
    
    if symptoms:
        context.append(f"User's reported symptoms: {symptoms}")
    if diseases:
        context.append(f"User's known conditions: {diseases}")
    
    return "\n".join(context)

def get_base_prompt():
    """Get the base system prompt without user context"""
    return """You are a medical chatbot assistant. Your role is to provide helpful medical information and guidance.

Key Guidelines:
1. Keep responses concise (maximum 3 lines, 500 characters)
2. Be professional and direct
3. Focus on factual medical information
4. Use  medical terminology than are common to general public
5. Include only essential disclaimers when necessary
6. When asked about the pdf, provide a brief summary of the content include the import details like positive negative points and the conclusion of the pdf make sure it is understandable to the user
7. Ask questions to clarify user needs

Medical Disclaimer:
- This is an AI assistant providing general information only. Consult healthcare providers for medical decisions."""

def get_system_prompt():
    """Get the system prompt with user's health context"""
    try:
        symptoms, diseases = get_user_health_info()
        health_context = format_health_context(symptoms, diseases)
        
        base_prompt = get_base_prompt()
        
        if health_context:
            base_prompt += f"\n\nUser's Health Context:\n{health_context}"
        
        return base_prompt
    except RuntimeError:
        return get_base_prompt()

def customize_response(response, symptoms=None, diseases=None):
    """Customize the response based on user's health information"""
    if not symptoms and not diseases:
        return response
    
    # Add relevant health context to the response
    if symptoms and any(symptom.lower() in response.lower() for symptom in symptoms.split(',')):
        response += "\nNote: Monitor these symptoms and consult your healthcare provider if they persist."
    
    if diseases and any(disease.lower() in response.lower() for disease in diseases.split(',')):
        response += "\nNote: Please follow your healthcare provider's recommendations for this condition."
    
    return response

# Initialize with base prompt
system_prompt = get_base_prompt()
