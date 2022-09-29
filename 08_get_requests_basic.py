# Testing Requests in Python

# import modules
import requests
import os
from PIL import Image
from IPython.display import IFrame

# Make a GET request via the method get to www.ibm.com:
r = requests.get('https://www.ibm.com')

# Print the status code of the response:
print(r.status_code)

# Print the request headers:
print(r.request.headers)

# print the request body (as there is no body for a get request we get a None):
print(r.request.body)

# print the response headers:
print(r.headers)

# print the date and content-type from the headers:
print("Date: " + r.headers['Date'])
print("Content-Type: " + r.headers['Content-Type'])

# print the encoding of the response:
print("Encoding: " + r.encoding)

# print the first 100 characters of the response:
print(r.text[:100])




# print a horizon line
print('-'*60)



# Use single quotation marks for defining string
url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/IDSNlogo.png'

# make a request to the url, and print the response headers:
r = requests.get(url)
print(r.headers)

# Using path.join Create a path for image.png based on the current directory:
path = os.path.join(os.getcwd(), 'image.png')

# print the path:
print(path)

# if the content-type is image/png then write the conent of the request to the path variable
if r.headers['Content-Type'] == 'image/png':
    with open(path, 'wb') as f:
        f.write(r.content)

# view the image (this doesn't work in VSCode for some reason):
img = Image.open(path)