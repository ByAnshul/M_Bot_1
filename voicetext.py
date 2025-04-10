import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Capture voice input from the microphone
with sr.Microphone() as source:
    print("Speak something...")
    recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
    audio = recognizer.listen(source)

try:
    # Convert audio to text
    text = recognizer.recognize_google(audio)
    print("You said: " + text)

except sr.UnknownValueError:
    print("Sorry, I could not understand your speech.")
except sr.RequestError:
    print("Sorry, the speech recognition service is unavailable.")
