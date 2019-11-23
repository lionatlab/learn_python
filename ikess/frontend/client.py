from requests import put, get

print(get('http://localhost:5000/', data={'data': 'Remember the milk'}).json())