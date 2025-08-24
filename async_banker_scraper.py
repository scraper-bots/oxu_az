#!/usr/bin/env python3
"""
Async Banker.az News Scraper

High-performance async scraper for banker.az with aiohttp optimization
and comprehensive content extraction with cleaning.
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, unquote
import re
from pathlib import Path
import csv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncBankerAzScraper:
    def __init__(self, concurrent_requests: int = 5, request_delay: float = 1.0):
        self.base_url = "https://banker.az"
        self.concurrent_requests = concurrent_requests
        self.request_delay = request_delay
        self.session = None
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page with rate limiting."""
        async with self.semaphore:
            try:
                logger.info(f"Fetching: {url}")
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'lxml')
                        await asyncio.sleep(self.request_delay)
                        return soup
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove unwanted characters and patterns
        text = re.sub(r'[\u200b-\u200f\u2060-\u206f]', '', text)  # Zero-width spaces
        text = re.sub(r'⚡️.*?WhatsApp.*?izlə', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'Təcili xəbərlər.*?izlə', '', text, flags=re.IGNORECASE)
        
        # Remove social media and promotional content
        text = re.sub(r'(Facebook|Twitter|WhatsApp|Telegram).*?paylaş', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Paylaş.*?(Facebook|Twitter|WhatsApp|Telegram)', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_news_list(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract news items from a category/list page with enhanced data."""
        news_items = []
        
        # Find all news items
        news_blocks = soup.find_all('div', class_='tdb_module_loop')
        
        for block in news_blocks:
            try:
                # Extract title and URL
                title_element = block.find('h3', class_='entry-title')
                if not title_element:
                    continue
                
                title_link = title_element.find('a')
                if not title_link:
                    continue
                
                title = self.clean_text(title_link.get_text())
                url = title_link.get('href')
                
                # Extract date with multiple fallbacks
                date_element = block.find('time', class_='entry-date')
                date_iso = None
                date_display = None
                
                if date_element:
                    date_iso = date_element.get('datetime')
                    date_display = self.clean_text(date_element.get_text())
                
                # Extract image URL with fallbacks
                image_url = None
                image_element = block.find('span', class_='entry-thumb')
                if image_element:
                    image_url = image_element.get('data-img-url')
                    if not image_url:
                        # Try background-image style
                        style = image_element.get('style', '')
                        bg_match = re.search(r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', style)
                        if bg_match:
                            image_url = bg_match.group(1)
                
                # Try to extract category if available
                category = None
                category_element = block.find('a', class_='tdb-entry-category')
                if category_element:
                    category = self.clean_text(category_element.get_text())
                
                # Extract excerpt/summary if available
                excerpt = None
                excerpt_element = block.find('div', class_='td-excerpt')
                if excerpt_element:
                    excerpt = self.clean_text(excerpt_element.get_text())
                
                news_item = {
                    'title': title,
                    'url': url,
                    'date_iso': date_iso,
                    'date_display': date_display,
                    'image_url': image_url,
                    'category': category,
                    'excerpt': excerpt
                }
                
                news_items.append(news_item)
                logger.debug(f"Extracted: {title}")
                
            except Exception as e:
                logger.warning(f"Error extracting news item: {e}")
                continue
        
        return news_items
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract comprehensive article content with enhanced cleaning."""
        article_data = {'url': url}
        
        try:
            # Extract title
            title_element = soup.find('h1', class_='tdb-title-text')
            article_data['title'] = self.clean_text(title_element.get_text()) if title_element else None
            
            # Extract date with multiple formats
            date_element = soup.find('time', class_='entry-date')
            if date_element:
                article_data['date_iso'] = date_element.get('datetime')
                article_data['date_display'] = self.clean_text(date_element.get_text())
            
            # Extract category
            category_element = soup.find('a', class_='tdb-entry-category')
            article_data['category'] = self.clean_text(category_element.get_text()) if category_element else None
            
            # Extract main image with fallbacks
            image_url = None
            image_containers = [
                soup.find('div', class_='tdb_single_featured_image'),
                soup.find('div', class_='td-post-featured-image'),
                soup.find('img', class_='entry-thumb')
            ]
            
            for container in image_containers:
                if container:
                    img = container.find('img') if container.name != 'img' else container
                    if img:
                        image_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                        if image_url and not image_url.startswith('data:'):
                            break
            
            article_data['image_url'] = image_url
            
            # Extract comprehensive content
            content_div = soup.find('div', class_='tdb_single_content')
            if content_div:
                content_inner = content_div.find('div', class_='tdb-block-inner')
                if content_inner:
                    # Remove unwanted elements
                    for unwanted in content_inner.find_all(['script', 'style', 'ins', 'iframe']):
                        unwanted.decompose()
                    
                    # Remove ads and promotional content
                    for ad in content_inner.find_all(['div'], class_=['td-a-ad', 'tdi_', 'g g-']):
                        ad.decompose()
                    
                    # Remove social sharing buttons
                    for social in content_inner.find_all('div', class_=['td-post-sharing', 'social-sharing']):
                        social.decompose()
                    
                    # Extract clean paragraphs
                    paragraphs = content_inner.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    clean_content = []
                    
                    for p in paragraphs:
                        text = self.clean_text(p.get_text())
                        if text and len(text) > 10:  # Filter out very short paragraphs
                            # Skip promotional/social media content
                            if not any(word in text.lower() for word in ['whatsapp', 'telegram', 'facebook', 'paylaş', '⚡️']):
                                clean_content.append(text)
                    
                    article_data['content'] = '\n\n'.join(clean_content)
                    
                    # Keep cleaned HTML for structure
                    article_data['content_html'] = str(content_inner)
            
            # Extract source
            source_element = soup.find('div', class_='tdb_single_source')
            if source_element:
                source_link = source_element.find('a')
                if source_link:
                    article_data['source'] = self.clean_text(source_link.get_text())
                else:
                    article_data['source'] = self.clean_text(source_element.get_text().replace('Mənbə', '').strip())
            
            # Extract tags
            tags_section = soup.find('div', class_='tdb_single_tags')
            if tags_section:
                tag_links = tags_section.find_all('a')
                article_data['tags'] = [self.clean_text(tag.get_text()) for tag in tag_links if tag.get_text().strip()]
            
            # Extract author if available
            author_element = soup.find('span', class_='td-author-name') or soup.find('div', class_='td-post-author-name')
            if author_element:
                article_data['author'] = self.clean_text(author_element.get_text())
            
            # Extract word count
            if article_data.get('content'):
                article_data['word_count'] = len(article_data['content'].split())
            
            # Extract reading time estimate
            if article_data.get('word_count'):
                article_data['reading_time_minutes'] = max(1, round(article_data['word_count'] / 200))
            
        except Exception as e:
            logger.error(f"Error extracting article content from {url}: {e}")
        
        return article_data
    
    async def scrape_category_page(self, url: str) -> List[Dict]:
        """Scrape a category page and return list of news items."""
        soup = await self.fetch_page(url)
        if not soup:
            return []
        
        return self.extract_news_list(soup)
    
    async def scrape_article(self, url: str) -> Dict:
        """Scrape an individual article and return its content."""
        soup = await self.fetch_page(url)
        if not soup:
            return {'url': url, 'error': 'Failed to fetch page'}
        
        return self.extract_article_content(soup, url)
    
    async def scrape_multiple_articles(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple articles concurrently."""
        tasks = [self.scrape_article(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error scraping {urls[i]}: {result}")
                articles.append({'url': urls[i], 'error': str(result)})
            else:
                articles.append(result)
        
        return articles
    
    async def scrape_multiple_pages(self, base_url: str, max_pages: int = 5) -> List[Dict]:
        """Scrape multiple pages from a category concurrently."""
        # Generate page URLs
        page_urls = []
        for page in range(1, max_pages + 1):
            if page == 1:
                page_urls.append(base_url)
            else:
                if base_url.endswith('/'):
                    page_urls.append(f"{base_url}page/{page}/")
                else:
                    page_urls.append(f"{base_url}/page/{page}/")
        
        # Scrape all pages concurrently
        tasks = [self.scrape_category_page(url) for url in page_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_news = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error scraping page {i+1}: {result}")
            elif result:  # Only add if not empty
                all_news.extend(result)
            else:
                logger.info(f"No content found on page {i+1}, might be end of content")
        
        return all_news
    
    def save_data(self, data: List[Dict], base_filename: str):
        """Save data in multiple formats."""
        # Save as JSON
        json_file = f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {json_file}")
        
        # Save as CSV
        if data:
            csv_file = f"{base_filename}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                # Get all possible fieldnames
                fieldnames = set()
                for item in data:
                    fieldnames.update(item.keys())
                fieldnames = sorted(list(fieldnames))
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in data:
                    # Convert lists to strings for CSV
                    csv_item = {}
                    for key, value in item.items():
                        if isinstance(value, list):
                            csv_item[key] = ', '.join(map(str, value))
                        else:
                            csv_item[key] = value
                    writer.writerow(csv_item)
            logger.info(f"Data saved to {csv_file}")


async def extract_all_data_from_news_page(url: str, max_pages: int = 10, max_articles: int = 100):
    """
    Comprehensive extraction of all data from a news category page.
    
    Args:
        url: Base URL of the news category
        max_pages: Maximum number of pages to scrape
        max_articles: Maximum number of articles to extract full content
    
    Returns:
        Dict with news list and detailed articles
    """
    
    async with AsyncBankerAzScraper(concurrent_requests=10, request_delay=0.5) as scraper:
        logger.info(f"Starting comprehensive extraction from: {url}")
        
        # Step 1: Extract all news items from multiple pages
        logger.info(f"Extracting news list from {max_pages} pages...")
        all_news = await scraper.scrape_multiple_pages(url, max_pages)
        logger.info(f"Found {len(all_news)} news items total")
        
        if not all_news:
            logger.warning("No news items found!")
            return {'news_list': [], 'articles': []}
        
        # Step 2: Extract full content for articles (limit for performance)
        articles_to_scrape = all_news[:max_articles]
        logger.info(f"Extracting full content for {len(articles_to_scrape)} articles...")
        
        article_urls = [item['url'] for item in articles_to_scrape if item.get('url')]
        detailed_articles = await scraper.scrape_multiple_articles(article_urls)
        
        # Step 3: Merge data
        articles_with_metadata = []
        for i, article in enumerate(detailed_articles):
            if i < len(articles_to_scrape):
                # Merge list metadata with full content
                merged = {**articles_to_scrape[i], **article}
                articles_with_metadata.append(merged)
            else:
                articles_with_metadata.append(article)
        
        # Step 4: Save comprehensive data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save news list
        scraper.save_data(all_news, f"banker_az_news_list_{timestamp}")
        
        # Save detailed articles
        scraper.save_data(articles_with_metadata, f"banker_az_articles_detailed_{timestamp}")
        
        # Create summary
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'source_url': url,
            'total_news_items': len(all_news),
            'detailed_articles': len(articles_with_metadata),
            'pages_scraped': max_pages,
            'categories': list(set(item.get('category') for item in all_news if item.get('category'))),
            'date_range': {
                'earliest': min((item.get('date_iso') for item in all_news if item.get('date_iso')), default=None),
                'latest': max((item.get('date_iso') for item in all_news if item.get('date_iso')), default=None)
            }
        }
        
        # Save summary
        with open(f"banker_az_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info("Comprehensive extraction completed!")
        logger.info(f"Summary: {len(all_news)} news items, {len(articles_with_metadata)} detailed articles")
        
        return {
            'news_list': all_news,
            'articles': articles_with_metadata,
            'summary': summary
        }


async def main():
    """Main function demonstrating the async scraper."""
    
    # Example URLs
    category_url = "https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/"
    sample_article = "https://banker.az/t%c9%99hsil-kreditind%c9%99-yenilik-bal-h%c9%99ddi-endirildi/"
    
    async with AsyncBankerAzScraper(concurrent_requests=10, request_delay=0.5) as scraper:
        
        # Demo 1: Quick category scrape
        logger.info("=== Demo 1: Quick category scrape ===")
        news_list = await scraper.scrape_category_page(category_url)
        logger.info(f"Found {len(news_list)} articles")
        
        # Demo 2: Sample article extraction
        logger.info("=== Demo 2: Sample article extraction ===")
        article = await scraper.scrape_article(sample_article)
        logger.info(f"Article: {article.get('title', 'N/A')}")
        logger.info(f"Content length: {len(article.get('content', ''))}")
        logger.info(f"Word count: {article.get('word_count', 0)}")
        
        # Demo 3: Multiple articles concurrently
        if news_list:
            logger.info("=== Demo 3: Multiple articles concurrently ===")
            sample_urls = [item['url'] for item in news_list[:3]]
            articles = await scraper.scrape_multiple_articles(sample_urls)
            logger.info(f"Scraped {len(articles)} articles concurrently")
        
        # Save demo results
        scraper.save_data(news_list, 'demo_news_list')
        scraper.save_data([article], 'demo_article_sample')
    
    # Demo 4: Comprehensive extraction
    logger.info("=== Demo 4: Comprehensive extraction ===")
    result = await extract_all_data_from_news_page(
        category_url, 
        max_pages=3, 
        max_articles=10
    )
    
    logger.info(f"Comprehensive extraction completed:")
    logger.info(f"- News items: {len(result['news_list'])}")
    logger.info(f"- Detailed articles: {len(result['articles'])}")


if __name__ == "__main__":
    # Update requirements
    print("Required packages:")
    print("pip install aiohttp beautifulsoup4 lxml")
    print("\nRunning scraper...")
    
    asyncio.run(main())