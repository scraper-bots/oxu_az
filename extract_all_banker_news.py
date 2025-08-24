#!/usr/bin/env python3
"""
SINGLE FILE COMPREHENSIVE BANKER.AZ NEWS EXTRACTOR

This script extracts ALL news data from banker.az with:
- Async/aiohttp optimization for high performance
- Clean content extraction (removes ads, social media, promotions)
- Multiple output formats (JSON, CSV)
- Comprehensive metadata extraction
- Automatic pagination handling

Usage:
    python3 extract_all_banker_news.py

Requirements:
    pip install aiohttp beautifulsoup4 lxml
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import logging
import re
import csv
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BankerAzExtractor:
    def __init__(self, concurrent_requests: int = 8, delay: float = 0.3):
        self.base_url = "https://banker.az"
        self.session = None
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.delay = delay
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'az,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=15)
        self.session = aiohttp.ClientSession(
            headers=self.headers, timeout=timeout, connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch page with rate limiting and error handling."""
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        await asyncio.sleep(self.delay)
                        return BeautifulSoup(content, 'lxml')
                    else:
                        logger.warning(f"HTTP {response.status}: {url}")
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning for Azerbaijani content."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u2060-\u206f]', '', text)
        
        # Remove promotional content patterns
        promo_patterns = [
            r'⚡️.*?WhatsApp.*?izlə',
            r'Təcili xəbərlər.*?izlə',
            r'WhatsApp kanalından.*?izlə',
            r'Facebook|Twitter|WhatsApp|Telegram.*?paylaş',
            r'Paylaş.*?Facebook|Twitter|WhatsApp|Telegram',
            r'Mənbə\s*:?\s*',
            r'Taglar\s*:?\s*'
        ]
        
        for pattern in promo_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()
    
    def extract_news_items(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract news items from category page."""
        items = []
        news_blocks = soup.find_all('div', class_='tdb_module_loop')
        
        for block in news_blocks:
            try:
                # Title and URL
                title_elem = block.find('h3', class_='entry-title')
                if not title_elem:
                    continue
                    
                link = title_elem.find('a')
                if not link:
                    continue
                
                title = self.clean_text(link.get_text())
                url = link.get('href')
                
                # Date
                date_elem = block.find('time', class_='entry-date')
                date_iso = date_elem.get('datetime') if date_elem else None
                date_display = self.clean_text(date_elem.get_text()) if date_elem else None
                
                # Image
                img_elem = block.find('span', class_='entry-thumb')
                image_url = img_elem.get('data-img-url') if img_elem else None
                
                # Category
                cat_elem = block.find('a', class_='tdb-entry-category')
                category = self.clean_text(cat_elem.get_text()) if cat_elem else None
                
                item = {
                    'title': title,
                    'url': url,
                    'date_iso': date_iso,
                    'date_display': date_display,
                    'image_url': image_url,
                    'category': category
                }
                items.append(item)
                
            except Exception as e:
                logger.debug(f"Error extracting news item: {e}")
                continue
        
        return items
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract full article content with comprehensive cleaning."""
        article = {'url': url}
        
        try:
            # Title
            title_elem = soup.find('h1', class_='tdb-title-text')
            article['title'] = self.clean_text(title_elem.get_text()) if title_elem else None
            
            # Date
            date_elem = soup.find('time', class_='entry-date')
            if date_elem:
                article['date_iso'] = date_elem.get('datetime')
                article['date_display'] = self.clean_text(date_elem.get_text())
            
            # Category
            cat_elem = soup.find('a', class_='tdb-entry-category')
            article['category'] = self.clean_text(cat_elem.get_text()) if cat_elem else None
            
            # Featured image
            img_containers = [
                soup.find('div', class_='tdb_single_featured_image'),
                soup.find('img', class_='entry-thumb')
            ]
            
            for container in img_containers:
                if container:
                    img = container.find('img') if container.name != 'img' else container
                    if img:
                        img_url = img.get('src') or img.get('data-src')
                        if img_url and not img_url.startswith('data:'):
                            article['image_url'] = img_url
                            break
            
            # Main content
            content_div = soup.find('div', class_='tdb_single_content')
            if content_div:
                content_inner = content_div.find('div', class_='tdb-block-inner')
                if content_inner:
                    # Remove unwanted elements
                    for elem in content_inner.find_all(['script', 'style', 'ins', 'iframe']):
                        elem.decompose()
                    
                    # Remove ads and social elements
                    for elem in content_inner.find_all('div', class_=lambda x: x and any(
                        cls in x for cls in ['td-a-ad', 'tdi_', 'g g-', 'social', 'sharing']
                    )):
                        elem.decompose()
                    
                    # Extract clean paragraphs
                    paragraphs = content_inner.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    clean_paragraphs = []
                    
                    for p in paragraphs:
                        text = self.clean_text(p.get_text())
                        # Filter meaningful content
                        if (text and 
                            len(text) > 15 and 
                            not any(word in text.lower() for word in [
                                'whatsapp', 'telegram', 'facebook', 'twitter', 
                                'paylaş', '⚡️', 'kanalından', 'izlə'
                            ])):
                            clean_paragraphs.append(text)
                    
                    article['content'] = '\n\n'.join(clean_paragraphs)
            
            # Source
            source_elem = soup.find('div', class_='tdb_single_source')
            if source_elem:
                source_link = source_elem.find('a')
                article['source'] = self.clean_text(source_link.get_text()) if source_link else None
            
            # Tags
            tags_elem = soup.find('div', class_='tdb_single_tags')
            if tags_elem:
                tag_links = tags_elem.find_all('a')
                article['tags'] = [self.clean_text(tag.get_text()) for tag in tag_links 
                                 if tag.get_text().strip() and tag.get_text().strip().lower() != 'taglar']
            
            # Content metrics
            if article.get('content'):
                words = article['content'].split()
                article['word_count'] = len(words)
                article['reading_time_minutes'] = max(1, round(len(words) / 200))
            
        except Exception as e:
            logger.error(f"Error extracting article content: {e}")
        
        return article
    
    async def scrape_category_pages(self, category_url: str, max_pages: int = 20) -> List[Dict]:
        """Scrape multiple category pages concurrently."""
        page_urls = []
        
        # Generate page URLs
        for page in range(1, max_pages + 1):
            if page == 1:
                page_urls.append(category_url)
            else:
                if category_url.endswith('/'):
                    page_urls.append(f"{category_url}page/{page}/")
                else:
                    page_urls.append(f"{category_url}/page/{page}/")
        
        # Fetch all pages
        tasks = [self.fetch(url) for url in page_urls]
        soups = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_news = []
        for i, soup in enumerate(soups):
            if isinstance(soup, Exception):
                logger.error(f"Error on page {i+1}: {soup}")
                continue
            elif soup:
                items = self.extract_news_items(soup)
                if items:
                    all_news.extend(items)
                    logger.info(f"Page {i+1}: Found {len(items)} items")
                else:
                    logger.info(f"Page {i+1}: No items found, stopping")
                    break
            else:
                logger.warning(f"Page {i+1}: Failed to fetch")
                break
        
        return all_news
    
    async def scrape_articles_content(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple articles concurrently."""
        tasks = []
        for url in urls:
            async def scrape_single(u):
                soup = await self.fetch(u)
                return self.extract_article_content(soup, u) if soup else {'url': u, 'error': 'Failed to fetch'}
            tasks.append(scrape_single(url))
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def save_data(self, data: List[Dict], filename: str):
        """Save data in JSON and CSV formats."""
        # JSON
        json_file = f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved {len(data)} items to {json_file}")
        
        # CSV
        if data:
            csv_file = f"{filename}.csv"
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                for item in data:
                    row = {}
                    for key, value in item.items():
                        if isinstance(value, list):
                            row[key] = '; '.join(map(str, value))
                        else:
                            row[key] = value
                    writer.writerow(row)
            logger.info(f"✅ Saved {len(data)} items to {csv_file}")


async def extract_all_banker_news(
    categories: List[str] = None,
    max_pages_per_category: int = 15,
    max_articles: int = 200
) -> Dict:
    """
    MAIN EXTRACTION FUNCTION
    
    Extract all news data from banker.az categories.
    
    Args:
        categories: List of category URLs. If None, uses default categories.
        max_pages_per_category: Max pages to scrape per category
        max_articles: Max articles to get full content for
    
    Returns:
        Dictionary with extracted data and summary
    """
    
    # Default categories if none provided
    if categories is None:
        categories = [
            "https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/",  # News
            "https://banker.az/category/iqtisadiyyat/",             # Economy
            "https://banker.az/category/mal-bazari/",               # Finance
            "https://banker.az/category/siyaset/",                  # Politics
        ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async with BankerAzExtractor() as extractor:
        logger.info("🚀 Starting comprehensive Banker.az extraction...")
        
        # Step 1: Extract news lists from all categories
        all_news_items = []
        
        for category_url in categories:
            logger.info(f"📰 Processing category: {category_url}")
            category_news = await extractor.scrape_category_pages(category_url, max_pages_per_category)
            
            # Add category info to each item
            category_name = category_url.split('/')[-2] if category_url.endswith('/') else category_url.split('/')[-1]
            for item in category_news:
                item['source_category'] = category_name
            
            all_news_items.extend(category_news)
            logger.info(f"✅ Category {category_name}: {len(category_news)} articles")
        
        # Remove duplicates based on URL
        unique_news = []
        seen_urls = set()
        for item in all_news_items:
            if item['url'] not in seen_urls:
                unique_news.append(item)
                seen_urls.add(item['url'])
        
        logger.info(f"📊 Total unique articles found: {len(unique_news)}")
        
        # Step 2: Extract full content for articles (limited for performance)
        articles_to_extract = unique_news[:max_articles]
        logger.info(f"📖 Extracting full content for {len(articles_to_extract)} articles...")
        
        article_urls = [item['url'] for item in articles_to_extract]
        detailed_articles = await extractor.scrape_articles_content(article_urls)
        
        # Merge news list data with detailed content
        final_articles = []
        for i, detailed in enumerate(detailed_articles):
            if isinstance(detailed, Exception):
                logger.error(f"Error processing article {i}: {detailed}")
                continue
            
            if i < len(articles_to_extract):
                merged = {**articles_to_extract[i], **detailed}
                final_articles.append(merged)
        
        # Step 3: Generate summary statistics
        categories_found = set(item.get('category') for item in final_articles if item.get('category'))
        sources_found = set(item.get('source') for item in final_articles if item.get('source'))
        dates_found = [item.get('date_iso') for item in final_articles if item.get('date_iso')]
        
        summary = {
            'extraction_timestamp': datetime.now().isoformat(),
            'total_news_items': len(unique_news),
            'detailed_articles': len(final_articles),
            'categories_processed': len(categories),
            'categories_found': sorted(list(categories_found)),
            'sources_found': sorted(list(sources_found)),
            'date_range': {
                'earliest': min(dates_found) if dates_found else None,
                'latest': max(dates_found) if dates_found else None
            },
            'content_stats': {
                'avg_word_count': sum(item.get('word_count', 0) for item in final_articles) / len(final_articles) if final_articles else 0,
                'total_words': sum(item.get('word_count', 0) for item in final_articles)
            }
        }
        
        # Step 4: Save all data
        logger.info("💾 Saving extracted data...")
        
        # Save news list
        extractor.save_data(unique_news, f"banker_news_list_{timestamp}")
        
        # Save detailed articles
        extractor.save_data(final_articles, f"banker_articles_detailed_{timestamp}")
        
        # Save summary
        with open(f"banker_extraction_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info("🎉 EXTRACTION COMPLETED!")
        logger.info(f"📈 Summary:")
        logger.info(f"   • Total news items: {summary['total_news_items']}")
        logger.info(f"   • Detailed articles: {summary['detailed_articles']}")
        logger.info(f"   • Categories: {len(summary['categories_found'])}")
        logger.info(f"   • Total words: {summary['content_stats']['total_words']:,}")
        
        return {
            'news_list': unique_news,
            'detailed_articles': final_articles,
            'summary': summary
        }


def main():
    """Main execution function."""
    
    print("🔥 BANKER.AZ COMPREHENSIVE NEWS EXTRACTOR")
    print("==========================================")
    print("This will extract all news data from banker.az")
    print("Including: titles, content, dates, categories, sources, tags")
    print("Output formats: JSON and CSV")
    print()
    
    # Configuration
    categories = [
        "https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/",      # News/Reports
        "https://banker.az/category/iqtisadiyyat/",                # Economy
        "https://banker.az/category/mal-bazari/",                  # Financial Market
        "https://banker.az/category/siyaset/",                     # Politics
        "https://banker.az/category/dovl%c9%99t/",                # Government
    ]
    
    max_pages = 10  # Per category
    max_articles = 150  # For detailed extraction
    
    print(f"Configuration:")
    print(f"• Categories: {len(categories)}")
    print(f"• Max pages per category: {max_pages}")
    print(f"• Max articles for detailed extraction: {max_articles}")
    print(f"• Estimated articles: ~{len(categories) * max_pages * 10}")
    print()
    
    # Run extraction
    try:
        result = asyncio.run(extract_all_banker_news(
            categories=categories,
            max_pages_per_category=max_pages,
            max_articles=max_articles
        ))
        
        print("\n✅ SUCCESS! Data has been saved to files.")
        print(f"Check the generated files:")
        print(f"• banker_news_list_*.json/csv - All news items")
        print(f"• banker_articles_detailed_*.json/csv - Full article content") 
        print(f"• banker_extraction_summary_*.json - Extraction statistics")
        
    except KeyboardInterrupt:
        print("\n⏸️  Extraction interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        logger.exception("Extraction failed")


if __name__ == "__main__":
    # Check requirements
    try:
        import aiohttp
        import lxml
    except ImportError:
        print("❌ Missing required packages!")
        print("Please install: pip install aiohttp beautifulsoup4 lxml")
        exit(1)
    
    main()