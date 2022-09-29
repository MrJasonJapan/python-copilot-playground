# Testing Requests with Querytring paramaeters in Python

# import modules
import requests
import os

url_get='http://httpbin.org/get'

# create a payload for name and id: 
payload = {'name': 'John', 'id': '123'}

# make a get request to the url with the payload:
r = requests.get(url_get, params=payload)

# print the url of the request:
print(r.url)

# print the status code of the request:
print(r.status_code)

# print the text of the request:
print(r.text)

# Check to see if response content-type is in the JSON format
if r.headers['Content-Type'] == 'application/json':
    # get the args key from the json:
    print(r.json()['args'])



