#!/usr/bin/env python3
"""
Banker.az News Scraper

This script scrapes news articles from banker.az website.
It can extract news from category pages and individual article pages.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BankerAzScraper:
    def __init__(self):
        self.base_url = "https://banker.az"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page."""
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_news_list(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract news items from a category/list page."""
        news_items = []
        
        # Find all news items
        news_blocks = soup.find_all('div', class_='tdb_module_loop')
        
        for block in news_blocks:
            try:
                # Extract title and URL
                title_link = block.find('h3', class_='entry-title').find('a')
                if not title_link:
                    continue
                
                title = title_link.get_text(strip=True)
                url = title_link.get('href')
                
                # Extract date
                date_element = block.find('time', class_='entry-date')
                date_str = date_element.get('datetime') if date_element else None
                date_display = date_element.get_text(strip=True) if date_element else None
                
                # Extract image URL
                image_element = block.find('span', class_='entry-thumb')
                image_url = None
                if image_element:
                    image_url = image_element.get('data-img-url')
                
                news_item = {
                    'title': title,
                    'url': url,
                    'date_iso': date_str,
                    'date_display': date_display,
                    'image_url': image_url
                }
                
                news_items.append(news_item)
                logger.info(f"Extracted: {title}")
                
            except Exception as e:
                logger.warning(f"Error extracting news item: {e}")
                continue
        
        return news_items
    
    def extract_article_content(self, soup: BeautifulSoup) -> Dict:
        """Extract full article content from an individual article page."""
        article_data = {}
        
        try:
            # Extract title
            title_element = soup.find('h1', class_='tdb-title-text')
            article_data['title'] = title_element.get_text(strip=True) if title_element else None
            
            # Extract date
            date_element = soup.find('time', class_='entry-date')
            article_data['date_iso'] = date_element.get('datetime') if date_element else None
            article_data['date_display'] = date_element.get_text(strip=True) if date_element else None
            
            # Extract category
            category_element = soup.find('a', class_='tdb-entry-category')
            article_data['category'] = category_element.get_text(strip=True) if category_element else None
            
            # Extract main image
            image_element = soup.find('div', class_='tdb_single_featured_image').find('img')
            article_data['image_url'] = image_element.get('src') if image_element else None
            
            # Extract article content
            content_div = soup.find('div', class_='tdb_single_content')
            if content_div:
                content_inner = content_div.find('div', class_='tdb-block-inner')
                if content_inner:
                    # Remove ads and unwanted elements
                    for ad in content_inner.find_all('div', class_='td-a-ad'):
                        ad.decompose()
                    
                    # Extract text paragraphs
                    paragraphs = content_inner.find_all('p')
                    content_text = []
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text and not text.startswith('⚡️') and 'WhatsApp' not in text:
                            content_text.append(text)
                    
                    article_data['content'] = '\n\n'.join(content_text)
                    article_data['content_html'] = str(content_inner)
            
            # Extract source
            source_element = soup.find('div', class_='tdb_single_source')
            if source_element:
                source_link = source_element.find('a')
                article_data['source'] = source_link.get_text(strip=True) if source_link else None
            
            # Extract tags
            tags_section = soup.find('div', class_='tdb_single_tags')
            if tags_section:
                tag_links = tags_section.find_all('a')
                article_data['tags'] = [tag.get_text(strip=True) for tag in tag_links]
            
        except Exception as e:
            logger.error(f"Error extracting article content: {e}")
        
        return article_data
    
    def scrape_category_page(self, url: str) -> List[Dict]:
        """Scrape a category page and return list of news items."""
        soup = self.get_page(url)
        if not soup:
            return []
        
        return self.extract_news_list(soup)
    
    def scrape_article(self, url: str) -> Dict:
        """Scrape an individual article and return its content."""
        soup = self.get_page(url)
        if not soup:
            return {}
        
        article_data = self.extract_article_content(soup)
        article_data['url'] = url
        
        return article_data
    
    def scrape_multiple_pages(self, base_url: str, max_pages: int = 5) -> List[Dict]:
        """Scrape multiple pages from a category."""
        all_news = []
        
        for page in range(1, max_pages + 1):
            if page == 1:
                # First page URL might not have /page/1/
                page_url = base_url
            else:
                # Construct page URL
                if base_url.endswith('/'):
                    page_url = f"{base_url}page/{page}/"
                else:
                    page_url = f"{base_url}/page/{page}/"
            
            logger.info(f"Scraping page {page}: {page_url}")
            news_items = self.scrape_category_page(page_url)
            
            if not news_items:
                logger.info(f"No news items found on page {page}, stopping")
                break
            
            all_news.extend(news_items)
            
            # Be respectful to the server
            time.sleep(1)
        
        return all_news
    
    def save_to_json(self, data: List[Dict], filename: str):
        """Save scraped data to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data saved to {filename}")

def main():
    """Main function to demonstrate the scraper."""
    scraper = BankerAzScraper()
    
    # Example 1: Scrape the category page provided
    category_url = "https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/page/2/"
    logger.info(f"Scraping category page: {category_url}")
    
    news_list = scraper.scrape_category_page(category_url)
    logger.info(f"Found {len(news_list)} news items")
    
    # Save news list
    scraper.save_to_json(news_list, 'banker_az_news_list.json')
    
    # Example 2: Scrape the individual article provided
    article_url = "https://banker.az/t%c9%99hsil-kreditind%c9%99-yenilik-bal-h%c9%99ddi-endirildi/"
    logger.info(f"Scraping article: {article_url}")
    
    article_data = scraper.scrape_article(article_url)
    logger.info(f"Article scraped: {article_data.get('title', 'Unknown')}")
    
    # Save article data
    scraper.save_to_json([article_data], 'banker_az_article_sample.json')
    
    # Example 3: Scrape first few articles from the list
    if news_list:
        logger.info("Scraping detailed content for first 3 articles...")
        detailed_articles = []
        
        for i, news_item in enumerate(news_list[:3]):
            logger.info(f"Scraping article {i+1}/3: {news_item['title']}")
            article_data = scraper.scrape_article(news_item['url'])
            detailed_articles.append(article_data)
            time.sleep(2)  # Be respectful
        
        scraper.save_to_json(detailed_articles, 'banker_az_detailed_articles.json')

if __name__ == "__main__":
    main()