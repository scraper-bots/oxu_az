#!/usr/bin/env python3
"""
COMPLETE BANKER.AZ NEWS EXTRACTOR - FIXED VERSION

This script extracts ALL news data with complete content and metadata:
- Full article content extraction
- Complete category information
- Proper image URLs (not placeholders)
- All metadata (source, tags, date, author)
- Clean content (removes ads, social media)

Usage: python3 complete_banker_extractor.py
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteBankerExtractor:
    def __init__(self, concurrent_requests: int = 6, delay: float = 0.5):
        self.base_url = "https://banker.az"
        self.session = None
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.delay = delay
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'az-AZ,az;q=0.9,en-US;q=0.8,en;q=0.7,tr;q=0.6',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Cache-Control': 'no-cache',
        }
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=45, connect=15)
        connector = aiohttp.TCPConnector(
            limit=50, 
            limit_per_host=20,
            ssl=False,  # Handle SSL issues
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=timeout,
            connector=connector,
            trust_env=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_with_retry(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch page with retry logic and better error handling."""
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Fetching (attempt {attempt + 1}): {url}")
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            soup = BeautifulSoup(content, 'lxml')
                            await asyncio.sleep(self.delay)
                            return soup
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    logger.error(f"Error fetching {url} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            
            logger.error(f"Failed to fetch {url} after {max_retries} attempts")
            return None
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning for Azerbaijani content."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove zero-width characters and invisible chars
        text = re.sub(r'[\u200b-\u200f\u2060-\u206f\ufeff]', '', text)
        
        # Remove promotional content patterns (more comprehensive)
        promo_patterns = [
            r'⚡️.*?WhatsApp.*?izlə',
            r'Təcili xəbərlər.*?izlə',
            r'WhatsApp kanalından.*?izlə',
            r'📲.*?WhatsApp.*?izlə',
            r'Facebook|Twitter|WhatsApp|Telegram.*?paylaş',
            r'Paylaş.*?Facebook|Twitter|WhatsApp|Telegram',
            r'Mənbə\s*:?\s*',
            r'Taglar\s*:?\s*',
            r'Paylaş\s*$',
            r'^Mənbə\s*',
            r'^Taglar\s*'
        ]
        
        for pattern in promo_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()
    
    def extract_news_items_enhanced(self, soup: BeautifulSoup) -> List[Dict]:
        """Enhanced news extraction with better category detection."""
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
                
                # Enhanced date extraction
                date_elem = block.find('time', class_='entry-date')
                date_iso = date_elem.get('datetime') if date_elem else None
                date_display = self.clean_text(date_elem.get_text()) if date_elem else None
                
                # Enhanced image extraction
                image_url = None
                
                # Try multiple image selectors
                image_selectors = [
                    block.find('span', class_='entry-thumb'),
                    block.find('div', class_='td-image-container'),
                    block.find('img'),
                ]
                
                for img_elem in image_selectors:
                    if img_elem:
                        if img_elem.name == 'span':
                            # Check data-img-url first
                            image_url = img_elem.get('data-img-url')
                            if not image_url:
                                # Try background-image style
                                style = img_elem.get('style', '')
                                bg_match = re.search(r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', style)
                                if bg_match:
                                    image_url = bg_match.group(1)
                        elif img_elem.name == 'img':
                            image_url = img_elem.get('src') or img_elem.get('data-src') or img_elem.get('data-lazy-src')
                        elif img_elem.name == 'div':
                            inner_img = img_elem.find('img')
                            if inner_img:
                                image_url = inner_img.get('src') or inner_img.get('data-src')
                        
                        if image_url and not image_url.startswith('data:'):
                            break
                
                # Enhanced category extraction - try multiple approaches
                category = None
                
                # Method 1: Direct category link in the block
                cat_elem = block.find('a', class_='tdb-entry-category')
                if cat_elem:
                    category = self.clean_text(cat_elem.get_text())
                
                # Method 2: Look for category in parent containers
                if not category:
                    parent = block.find_parent('div', class_='td-category-pos-above')
                    if parent:
                        cat_link = parent.find('a')
                        if cat_link:
                            category = self.clean_text(cat_link.get_text())
                
                # Method 3: Check for category in URL pattern
                if not category and url:
                    url_match = re.search(r'/category/([^/]+)/', url)
                    if url_match:
                        category_slug = url_match.group(1)
                        # Convert URL slug to readable name
                        category = category_slug.replace('%c9%99', 'ə').replace('-', ' ').title()
                
                # Extract excerpt/summary if available
                excerpt = None
                excerpt_elem = block.find('div', class_='td-excerpt')
                if excerpt_elem:
                    excerpt = self.clean_text(excerpt_elem.get_text())
                
                item = {
                    'title': title,
                    'url': url,
                    'date_iso': date_iso,
                    'date_display': date_display,
                    'image_url': image_url,
                    'category': category,
                    'excerpt': excerpt,
                    'extracted_from': 'news_list'
                }
                
                items.append(item)
                logger.debug(f"Extracted: {title} | Category: {category}")
                
            except Exception as e:
                logger.warning(f"Error extracting news item: {e}")
                continue
        
        return items
    
    def extract_complete_article_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract complete article content with ALL metadata."""
        article = {'url': url, 'extracted_from': 'article_page'}
        
        try:
            # Title
            title_elem = soup.find('h1', class_='tdb-title-text')
            article['title'] = self.clean_text(title_elem.get_text()) if title_elem else None
            
            # Enhanced date extraction
            date_elem = soup.find('time', class_='entry-date')
            if date_elem:
                article['date_iso'] = date_elem.get('datetime')
                article['date_display'] = self.clean_text(date_elem.get_text())
            
            # Enhanced category extraction with multiple fallbacks
            category = None
            
            # Method 1: tdb-entry-category
            cat_elem = soup.find('a', class_='tdb-entry-category')
            if cat_elem:
                category = self.clean_text(cat_elem.get_text())
            
            # Method 2: tdb_single_categories
            if not category:
                cat_container = soup.find('div', class_='tdb_single_categories')
                if cat_container:
                    cat_link = cat_container.find('a')
                    if cat_link:
                        category = self.clean_text(cat_link.get_text())
            
            # Method 3: breadcrumbs or navigation
            if not category:
                breadcrumbs = soup.find('nav', class_='breadcrumb') or soup.find('div', class_='breadcrumb')
                if breadcrumbs:
                    links = breadcrumbs.find_all('a')
                    if len(links) > 1:  # Skip "Home" link
                        category = self.clean_text(links[-2].get_text())
            
            article['category'] = category
            
            # Enhanced featured image extraction
            image_url = None
            
            # Try multiple image containers
            image_containers = [
                soup.find('div', class_='tdb_single_featured_image'),
                soup.find('div', class_='td-post-featured-image'),
                soup.find('figure', class_='wp-block-image'),
                soup.find('img', class_='entry-thumb'),
                soup.find('img', class_='wp-post-image'),
            ]
            
            for container in image_containers:
                if container:
                    img = container.find('img') if container.name != 'img' else container
                    if img:
                        # Try multiple src attributes
                        img_url = (img.get('src') or 
                                 img.get('data-src') or 
                                 img.get('data-lazy-src') or
                                 img.get('data-original'))
                        
                        if img_url and not img_url.startswith('data:'):
                            image_url = img_url
                            break
            
            article['image_url'] = image_url
            
            # COMPLETE content extraction
            content_div = soup.find('div', class_='tdb_single_content')
            if content_div:
                content_inner = content_div.find('div', class_='tdb-block-inner')
                if content_inner:
                    # Make a copy to avoid modifying original
                    content_copy = BeautifulSoup(str(content_inner), 'lxml')
                    
                    # Remove unwanted elements more comprehensively
                    unwanted_selectors = [
                        'script', 'style', 'ins', 'iframe', 'noscript',
                        'div[class*="td-a-ad"]',
                        'div[class*="tdi_"]',
                        'div[class*="g g-"]',
                        'div[class*="social"]',
                        'div[class*="sharing"]',
                        'div[class*="adviad"]',
                        'div[id*="adviad"]',
                        '.td-post-sharing',
                        '.social-sharing',
                        '.advertisement',
                        '.ad-banner'
                    ]
                    
                    for selector in unwanted_selectors:
                        for elem in content_copy.select(selector):
                            elem.decompose()
                    
                    # Extract ALL text content - paragraphs, headings, lists
                    content_elements = content_copy.find_all([
                        'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                        'ul', 'ol', 'li', 'blockquote', 'div'
                    ])
                    
                    clean_content = []
                    
                    for elem in content_elements:
                        text = self.clean_text(elem.get_text())
                        
                        # More lenient filtering - keep meaningful content
                        if (text and 
                            len(text) > 10 and 
                            not any(skip_word in text.lower() for skip_word in [
                                'whatsapp kanalından', 'telegram', 'paylaş', 
                                '⚡️', 'təcili xəbərlər', 'facebook', 'twitter'
                            ])):
                            
                            # Add element type prefix for structure
                            if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                                clean_content.append(f"\n## {text}\n")
                            elif elem.name == 'blockquote':
                                clean_content.append(f"\n> {text}\n")
                            elif elem.name in ['ul', 'ol']:
                                # Handle lists
                                items = elem.find_all('li')
                                for li in items:
                                    li_text = self.clean_text(li.get_text())
                                    if li_text:
                                        clean_content.append(f"• {li_text}")
                            elif elem.name == 'li':
                                # Skip if already handled by parent ul/ol
                                continue
                            else:
                                clean_content.append(text)
                    
                    # Join content with proper spacing
                    full_content = '\n\n'.join(clean_content).strip()
                    article['content'] = full_content
                    
                    # Keep original HTML (cleaned)
                    article['content_html'] = str(content_copy)
            
            # Enhanced source extraction
            source_elem = soup.find('div', class_='tdb_single_source')
            if source_elem:
                source_link = source_elem.find('a')
                if source_link:
                    article['source'] = self.clean_text(source_link.get_text())
                else:
                    # Extract from text content
                    source_text = self.clean_text(source_elem.get_text())
                    article['source'] = source_text.replace('Mənbə', '').replace(':', '').strip()
            
            # Enhanced tags extraction
            tags_elem = soup.find('div', class_='tdb_single_tags')
            if tags_elem:
                tag_links = tags_elem.find_all('a')
                tags = []
                for tag in tag_links:
                    tag_text = self.clean_text(tag.get_text())
                    if tag_text and tag_text.lower() not in ['taglar', 'tags']:
                        tags.append(tag_text)
                article['tags'] = tags if tags else None
            
            # Author extraction
            author_selectors = [
                soup.find('span', class_='td-author-name'),
                soup.find('div', class_='td-post-author-name'),
                soup.find('a', rel='author'),
                soup.find('span', class_='author'),
            ]
            
            for author_elem in author_selectors:
                if author_elem:
                    author = self.clean_text(author_elem.get_text())
                    if author:
                        article['author'] = author
                        break
            
            # Content metrics
            if article.get('content'):
                words = article['content'].split()
                article['word_count'] = len(words)
                article['character_count'] = len(article['content'])
                article['reading_time_minutes'] = max(1, round(len(words) / 200))
                
                # Paragraph count
                paragraphs = [p for p in article['content'].split('\n\n') if p.strip()]
                article['paragraph_count'] = len(paragraphs)
            
            # Publication info
            article['extraction_date'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error extracting article content from {url}: {e}")
            article['extraction_error'] = str(e)
        
        return article
    
    async def scrape_category_pages(self, category_url: str, max_pages: int = 6800) -> List[Dict]:
        """Scrape multiple category pages with enhanced extraction."""
        page_urls = []
        
        for page in range(1, max_pages + 1):
            if page == 1:
                page_urls.append(category_url)
            else:
                if category_url.endswith('/'):
                    page_urls.append(f"{category_url}page/{page}/")
                else:
                    page_urls.append(f"{category_url}/page/{page}/")
        
        # Fetch pages with better concurrency control
        all_news = []
        batch_size = 8  # Process in larger batches for speed
        
        for i in range(0, len(page_urls), batch_size):
            batch_urls = page_urls[i:i + batch_size]
            tasks = [self.fetch_with_retry(url) for url in batch_urls]
            soups = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, soup in enumerate(soups):
                page_num = i + j + 1
                
                if isinstance(soup, Exception):
                    logger.error(f"Page {page_num} error: {soup}")
                    continue
                elif soup:
                    items = self.extract_news_items_enhanced(soup)
                    if items:
                        all_news.extend(items)
                        logger.info(f"Page {page_num}: Found {len(items)} items")
                    else:
                        logger.info(f"Page {page_num}: No items found, stopping pagination")
                        return all_news  # Stop if no items found
                else:
                    logger.warning(f"Page {page_num}: Failed to fetch")
        
        return all_news
    
    async def scrape_articles_complete(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple articles with complete content extraction."""
        batch_size = 10  # Larger batches for faster article extraction
        all_articles = []
        
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            logger.info(f"Processing articles batch {i//batch_size + 1}: {i+1}-{min(i+batch_size, len(urls))} of {len(urls)}")
            
            tasks = []
            for url in batch_urls:
                async def scrape_single(u):
                    soup = await self.fetch_with_retry(u)
                    if soup:
                        return self.extract_complete_article_content(soup, u)
                    else:
                        return {'url': u, 'error': 'Failed to fetch', 'extracted_from': 'article_page'}
                
                tasks.append(scrape_single(url))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Article extraction error: {result}")
                else:
                    all_articles.append(result)
        
        return all_articles
    
    def save_single_json(self, data: Dict, filename: str):
        """Save all data to a single JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
        logger.info(f"✅ Saved all data to {filename}")


async def extract_all_banker_news_complete():
    """Main extraction function with complete data extraction."""
    
    print("🚀 COMPLETE BANKER.AZ NEWS EXTRACTOR")
    print("=" * 50)
    print("This will extract COMPLETE news data including:")
    print("• Full article content")
    print("• Complete metadata (category, tags, source, author)")
    print("• Proper image URLs")
    print("• Clean content (no ads/promotions)")
    print("• MAXIMUM SCRAPING - all available pages and articles")
    print()
    
    # Single category to scrape - ALL pages, ALL articles
    single_category_url = "https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/"
    
    max_pages = 6800   # NO LIMIT - will auto-stop when no more content found
    max_articles = 1000000   # NO LIMIT - extract ALL articles found
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async with CompleteBankerExtractor() as extractor:
        logger.info("🚀 Starting COMPLETE extraction...")
        
        # Step 1: Extract ALL news from single category
        logger.info(f"📰 Processing single category: {single_category_url}")
        all_news_items = await extractor.scrape_category_pages(single_category_url, max_pages)
        
        # Add source category info
        category_name = single_category_url.split('/')[-2] if single_category_url.endswith('/') else single_category_url.split('/')[-1]
        for item in all_news_items:
            item['source_category_url'] = single_category_url
            item['source_category_name'] = category_name
        
        logger.info(f"✅ {category_name}: {len(all_news_items)} articles")
        
        # Remove duplicates
        unique_news = []
        seen_urls = set()
        for item in all_news_items:
            if item['url'] not in seen_urls:
                unique_news.append(item)
                seen_urls.add(item['url'])
        
        logger.info(f"📊 Total unique articles found: {len(unique_news)}")
        
        # Step 2: Extract COMPLETE content for ALL articles (no limit)
        logger.info(f"📖 Extracting COMPLETE content for ALL {len(unique_news)} articles...")
        
        article_urls = [item['url'] for item in unique_news]
        complete_articles = await extractor.scrape_articles_complete(article_urls)
        
        # Merge list data with complete content
        final_articles = []
        for i, complete_article in enumerate(complete_articles):
            if i < len(unique_news):
                # Merge list data with article data (article data takes priority)
                merged = {**unique_news[i], **complete_article}
                final_articles.append(merged)
        
        # Step 3: Generate summary and create single comprehensive file
        logger.info("💾 Saving ALL data to single JSON file...")
        
        categories_found = set(item.get('category') for item in final_articles if item.get('category'))
        sources_found = set(item.get('source') for item in final_articles if item.get('source'))
        authors_found = set(item.get('author') for item in final_articles if item.get('author'))
        
        # Create single comprehensive JSON file with all data
        comprehensive_data = {
            'extraction_info': {
                'extraction_timestamp': datetime.now().isoformat(),
                'total_news_items': len(unique_news),
                'complete_articles': len(final_articles),
                'single_category_processed': single_category_url,
                'categories_found': sorted(list(categories_found)),
                'sources_found': sorted(list(sources_found)),
                'authors_found': sorted(list(authors_found)),
                'content_stats': {
                    'avg_word_count': sum(item.get('word_count', 0) for item in final_articles) / len(final_articles) if final_articles else 0,
                    'total_words': sum(item.get('word_count', 0) for item in final_articles),
                    'avg_reading_time': sum(item.get('reading_time_minutes', 0) for item in final_articles) / len(final_articles) if final_articles else 0,
                },
                'extraction_config': {
                    'max_pages': max_pages,
                    'max_articles': max_articles,
                    'single_category_scraped': single_category_url
                }
            },
            'news_list': unique_news,
            'articles_with_content': final_articles
        }
        
        # Save single JSON file
        output_filename = f"banker_az_complete_data_{timestamp}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, ensure_ascii=False, indent=2, sort_keys=False)
        
        logger.info(f"✅ All data saved to single file: {output_filename}")
        summary = comprehensive_data['extraction_info']
        
        logger.info("🎉 COMPLETE EXTRACTION FINISHED!")
        logger.info(f"📈 Results:")
        logger.info(f"   • Total news items: {summary['total_news_items']}")
        logger.info(f"   • Complete articles: {summary['complete_articles']}")
        logger.info(f"   • Categories found: {len(summary['categories_found'])}")
        logger.info(f"   • Sources found: {len(summary['sources_found'])}")
        logger.info(f"   • Total words extracted: {summary['content_stats']['total_words']:,}")
        logger.info(f"   • Average article length: {summary['content_stats']['avg_word_count']:.0f} words")
        
        return comprehensive_data


if __name__ == "__main__":
    # Check requirements
    try:
        import aiohttp
        import lxml
    except ImportError:
        print("❌ Missing packages! Install: pip install aiohttp beautifulsoup4 lxml")
        exit(1)
    
    # Run complete extraction
    try:
        asyncio.run(extract_all_banker_news_complete())
        print("\n✅ SUCCESS! Check the generated file:")
        print("• banker_az_complete_data_*.json - Single JSON file with all data")
        
    except KeyboardInterrupt:
        print("\n⏸️  Extraction interrupted")
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        logger.exception("Complete extraction failed")