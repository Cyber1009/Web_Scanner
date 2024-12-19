import requests
from bs4 import BeautifulSoup
# from fake_useragent import UserAgent

# ua = UserAgent()

# headers = {'User-Agent' : ua.random}

# Send a GET request to the website
url = "https://www.tcyoung.co.uk/"
response = requests.get(url, headers = {"User-Agent":"Mozilla/5.0"}, timeout= 10)
print('re:', response)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the data you want
# Example: Extract all the titles
titles = soup.find_all('h2')
for title in titles:
    print(title.text)