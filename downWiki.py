import datetime
import requests
import json
import threading
import webbrowser
import urllib.parse

from http.server import BaseHTTPRequestHandler, HTTPServer

"""
This python script demonstrates how to implement the OAUTH2 flow where:
1. a user loads a browser url which logs into drchrono.
2. drchrono will prompt the user to authorize a custom web application to operate using their credentials.
3. once the user authorizes the action they will be redirected to the web application with an authorization code.
4. the web application will receive the request containing the authorization code.
5. the web application will exchange the authorization code for an API token.
6. the web application will use the API token to retrieve a list of appointments on behalf of the user.

This script requires python runtime 3.6+ and python "requests" library
Always keep in mind client_secret and access_token are sensitive information and subject to HIPAA least-use policy

Example output:
c:\temp>python.exe thescript.py
Starting web server at http://localhost:8000
https://drchrono.com/o/authorize/?scope=calendar%3Aread%20patients%3Aread%20clinical%3Aread&response_type=code&redirect_uri=http://localhost:8000&client_id=XXX
Received GET with URL /?code=YYY
Found auth code YYY
POSTing to token endpoint using client-id/client-secret/auth-code/redirect-uri to get a token
Server responded with {"access_token": "ZZZ", "token_type": "Bearer", "expires_in": 172800, "refresh_token": "AAA", "scope": "calendar:read patients:read clinical:read"}
Token expires at 2024-04-18 09:15:29
Using the token to load appointments
Server responded with {"previous":null,"results":[{...}]}
"""


# this should be a URL to a web server which you control.
# when the user logs in to drchrono they will be redirected to this location with the authorization_code in the querystring.
# this authorization code can be used to log on to the API as the user.
redirect_uri = 'http://localhost:8000'
# you must set up an API application inside drchrono to retrieve these values
client_id = 'DRCHRONO_PROVIDED'
client_secret = 'DRCHRONO_PROVIDED'


class MyWebServer(BaseHTTPRequestHandler):
    """ Replace this with your web application listening on a specific URL """
    def do_GET(self):
        """ listening for a GET request which happens after user logs on and is redirected """
        try:
            print(f'Received GET request with URL {self.path}')
            authorization_code = self.path.split('code=')[1]
            print(f'Found auth code {authorization_code}')
            print(f'POSTing to token endpoint using client-id/client-secret/auth-code/redirect-uri to get a token')
            response = requests.post(f'https://drchrono.com/o/token/',
                params={
                    'grant_type': 'authorization_code',
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'redirect_uri': redirect_uri,
                    'code': authorization_code,
                },
            )
            response.raise_for_status()
            print(f'Server responded with {response.text}')
            token = response.json()
            expires_in = datetime.datetime.now() + datetime.timedelta(seconds=token['expires_in'])
            print(f'Token expires at {expires_in.strftime("%Y-%m-%d %H:%M:%S")}')
            print(f'Using the token to load appointments')
            response = requests.get(
                f'https://drchrono.com/api/appointments?since=2024-01-01',
                headers={
                    'Authorization': f'Bearer {token["access_token"]}',
                    'Content-Type': 'application/json',
                },
            )
            response.raise_for_status()
            print(f'Server responded with {response.text}')
        finally:
            raise SystemExit


print(f'Starting disposable web server at {redirect_uri}')
server_thread = threading.Thread(target=lambda: HTTPServer(('localhost', 8000), MyWebServer).serve_forever())
server_thread.start()

# scopes allow you as the client to limit the capabilities the API token will have
permitted_scopes = ['calendar:read', 'patients:read', 'clinical:read']
scope_string = urllib.parse.quote(" ".join(permitted_scopes), safe='')
# this is a url link which the user would click on to initiate the OAUTH2 process
browser_url = f'https://drchrono.com/o/authorize/?scope={scope_string}&response_type=code&redirect_uri={redirect_uri}&client_id={client_id}'
webbrowser.open(browser_url, new=2)

# the web server awaits the user to perform the logon process and get redirected back to the web server with the auth token
server_thread.join()
 

