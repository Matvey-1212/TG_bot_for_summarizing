import requests
from bs4 import BeautifulSoup
import json

#Destination page URL
url = "https://lenta.ru/rss/google-newsstand/main/"

#Send an HTTP request to retrieve web content
response = requests.get(url)
if response.status_code != 200:
    print(f"Request failed, status code: {response.status_code}")
    exit()

# Parse web content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# List for storing news data
news_data = []

# Iterate through each news item
for item in soup.find_all('item'):
    title = item.find('title').text
    link = item.find('link').text
    ##pub_date = item.find('pubDate').text
    category = item.find('category').text if item.find('category') else '无分类'
    creator = item.find('dc:creator').text if item.find('dc:creator') else '无作者'
    description = item.find('description').text if item.find('description') else '无描述'

    # Building a news data dictionary
    news_item = {
        'title': title,
        'link': link,
        ##'pub_date': pub_date,
        'category': category,
        'creator': creator,
        'description': description
    }

    news_data.append(news_item)

# Save news data as a JSON file
with open('news_dataset.json', 'w', encoding='utf-8') as file:
    json.dump(news_data, file, ensure_ascii=False, indent=4)

print("新闻数据已成功抓取并保存为news_dataset.json")