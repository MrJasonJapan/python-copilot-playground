# Testing post requests in Python

# import modules
import requests
import os

url_post='http://httpbin.org/post'

# create a payload for name and id: 
payload = {'name': 'John', 'id': '123'}

# make a post request to the url with the payload:
r_post = requests.post(url_post, data=payload)

# print the url of the request: (notice how we don't see the payload in the querystring)
print(r_post.url)

# print the body of the response: (notice how we see the payload sent back to us in the "form" key.)
print(r_post.text)

# print just the form of the response:
print(r_post.json()['form'])

# print the resonse status code:
print(r_post.status_code)
