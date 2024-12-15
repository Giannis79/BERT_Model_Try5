import requests
from bs4 import BeautifulSoup
import csv
import time
import random

def get_articles_from_sitemap(sitemap_url, keywords):
    # Step 1: Fetch the XML file
    response = requests.get(sitemap_url)
    if response.status_code != 200:
        print(f"Failed to retrieve the sitemap from {sitemap_url}")
        return []

    # Step 2: Parse the XML content
    soup = BeautifulSoup(response.content, 'lxml-xml')
    article_data = []

    # Step 3: Filter URLs by keywords
    for url in soup.find_all('url'):
        loc = url.find('loc').text

        # Print URLs for debugging
        print(f"URL: {loc}")

        if any(keyword in loc.lower() for keyword in keywords):
            # Step 4: Fetch and process article content
            article_text = get_article_text(loc)
            if article_text:
                # Generate a title from the URL
                title = loc.split('/')[-1].replace('-', ' ').replace('.html', '').title()
                article_data.append((title, loc, article_text))
            # Respectful crawling by adding a delay
            time.sleep(random.uniform(1, 3))

    return article_data

def get_article_text(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve article text from {url}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ' '.join([para.get_text().strip() for para in paragraphs])

    return article_text

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "URL", "Content"])
        writer.writerows(data)

# Example usage:
sitemap_url = "https://www.aljazeera.com/sitemaps/article-archive/2022/08.xml"
keywords = ["russia", "ukraine", "putin", "zelensky"]
articles = get_articles_from_sitemap(sitemap_url, keywords)

if articles:
    print(f"Found {len(articles)} articles containing keywords")
    save_to_csv(articles, 'filtered_articles_2022_08.csv')
    print("Articles saved to 'filtered_articles_2022_08.csv'")
else:
    print("No articles found.")