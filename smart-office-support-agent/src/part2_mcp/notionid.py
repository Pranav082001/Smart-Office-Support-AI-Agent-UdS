import os
from notion_client import Client

notion = Client(auth=os.environ.get("NOTION_TOKEN"))

# 搜索所有数据库
results = notion.search(filter={"property": "object", "value": "database"})

for item in results.get("results", []):
    print(f"Title: {item['title'][0]['plain_text'] if item['title'] else 'Untitled'}")
    print(f"ID: {item['id']}")
    print("---")