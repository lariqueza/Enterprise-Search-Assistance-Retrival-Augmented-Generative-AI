from atlassian import Confluence
from bs4 import BeautifulSoup
from textblob import TextBlob
import requests
from requests.auth import HTTPBasicAuth
import json
from sentence_transformers import SentenceTransformer
import db


#create instance for text data embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#connect to confluence
url = 'https://fellowshipai.atlassian.net'
spaces_API = 'https://fellowshipai.atlassian.net/wiki/api/v2/spaces'
user_name = ''
password_token = ''

#get spaces from confluence
authentication = HTTPBasicAuth(user_name, password_token)
headers = {
    "Accept": "application/json"
}

#call spaces API and store response
response = requests.request(
    "GET",
    spaces_API,
    headers=headers,
    auth=authentication
)

#store response in a json file
json_data = json.loads(response.text) 
print(json_data)

#loop through json file and extract space info
space_data = {}
for result in json_data['results']:
    space_data[result['key']] = {result['name']}

for key, value in space_data.items():
    print(f"Space Id  {key}:, Space Name {value}")


#get page data from space name
confluence = Confluence(
    url = 'https://fellowshipai.atlassian.net',
    username = 'okiajibola0@gmail.com',
    password = password_token
)
SPACE_KEY = 'FD'
pages = confluence.get_all_pages_from_space(SPACE_KEY, start=0, limit=50, expand='body.storage')
for page in pages:
    print(f"page id: {page['id']}, page title: {page['title']}")

def get_confluence_pages(confluence, SPACE_KEY, start=0, limit=50,):
    pages = confluence.get_all_pages_from_space(SPACE_KEY, start=start, limit=limit, expand='body.storage')

#extract data function
def text_extraction(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n')

#grammar correction
def correction(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

#get page by id extract data, clean data, embed data and insert into database
db.collection.drop()
PAGE_IDs = ['1507335', '1540120', '1900547', '2031719', '2097153', '2097163']
for PAGE_ID in PAGE_IDs:
    page = confluence.get_page_by_id(PAGE_ID, expand='body.storage')
    content = page['body']['storage']['value']
    title = page['title']
    first_output = text_extraction(content)
    clean_output = correction(first_output)
    embedded_output = model.encode(clean_output)
    db.collection.insert_one({"title": title, "content": clean_output, "embedding": embedded_output.tolist()})
    print("data inserted successfully")


