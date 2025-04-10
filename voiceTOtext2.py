from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import requests
import json

app = Flask(__name__)
CORS(app)

# Replace this with your Google API Key
API_KEY = "AIzaSyAcyY1XoBCNg8qcSYk9oDeChC40-PzkevA"

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    try:
        audio_data = request.json['audio_data']
        audio_content = base64.b64decode(audio_data)
        audio_base64 = base64.b64encode(audio_content).decode("utf-8")

        url = f"https://speech.googleapis.com/v1/speech:recognize?key={API_KEY}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "config": {
                "encoding": "WEBM_OPUS",
                "sampleRateHertz": 48000,
                "languageCode": "en-US"
            },
            "audio": {
                "content": audio_base64
            }
        }

        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if "results" in response_data:
            text = response_data['results'][0]['alternatives'][0]['transcript']
            return jsonify({"text": text})
        else:
            return jsonify({"error": response_data.get('error', {}).get('message', 'Error processing audio')})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
