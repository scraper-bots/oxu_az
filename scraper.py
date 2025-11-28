import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import json
from datetime import datetime
import logging
from pathlib import Path
import time
import sys


# Configure logging with UTF-8 encoding
file_handler = logging.FileHandler('scraper.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Configure console handler with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
# Set UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # For older Python versions, create a new StreamHandler with UTF-8
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)


class OxuAzScraper:
    """Crash-proof async scraper for oxu.az news website"""

    def __init__(
        self,
        start_page: int = 11,
        end_page: int = 10000,
        max_retries: int = 3,
        retry_delay: int = 2,
        request_delay: float = 0.5,
        timeout: int = 30
    ):
        self.start_page = start_page
        self.end_page = end_page
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.timeout = timeout
        self.base_url = "https://oxu.az/page/{}"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.articles = []
        self.failed_pages = []
        self.checkpoint_file = 'scraper_checkpoint.json'

    async def fetch_page(self, session: aiohttp.ClientSession, page_number: int) -> Optional[str]:
        """Fetch a single page with retry logic"""
        url = self.base_url.format(page_number)

        for attempt in range(1, self.max_retries + 1):
            try:
                # Add delay to avoid rate limiting
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)

                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.get(url, headers=self.headers, timeout=timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"✓ Successfully fetched page {page_number}")
                        return content
                    elif response.status == 403:
                        logger.warning(f"✗ Access forbidden for page {page_number} (403)")
                        self.failed_pages.append({'page': page_number, 'error': 'Access Forbidden'})
                        return None
                    elif response.status == 404:
                        logger.warning(f"✗ Page {page_number} not found (404)")
                        self.failed_pages.append({'page': page_number, 'error': 'Not Found'})
                        return None
                    else:
                        logger.warning(f"✗ Error fetching page {page_number}: Status {response.status} (Attempt {attempt}/{self.max_retries})")

            except asyncio.TimeoutError:
                logger.warning(f"✗ Timeout fetching page {page_number} (Attempt {attempt}/{self.max_retries})")
            except aiohttp.ClientError as e:
                logger.warning(f"✗ Client error fetching page {page_number}: {str(e)} (Attempt {attempt}/{self.max_retries})")
            except Exception as e:
                logger.error(f"✗ Unexpected error fetching page {page_number}: {str(e)} (Attempt {attempt}/{self.max_retries})")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait_time = self.retry_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying page {page_number} in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

        # All retries failed
        self.failed_pages.append({'page': page_number, 'error': 'Max retries exceeded'})
        logger.error(f"✗ Failed to fetch page {page_number} after {self.max_retries} attempts")
        return None

    def parse_article(self, article_elem) -> Optional[Dict]:
        """Parse a single article element with error handling"""
        try:
            article_data = {}

            # Extract article URL
            try:
                url_elem = article_elem.find('a', href=True)
                article_url = url_elem['href'] if url_elem else None
                if article_url and not article_url.startswith('http'):
                    article_url = f"https://oxu.az{article_url}"
                article_data['url'] = article_url
            except Exception as e:
                logger.debug(f"Error extracting URL: {str(e)}")
                article_data['url'] = None

            # Extract title
            try:
                title_elem = article_elem.find('h2', class_='post-item-title')
                article_data['title'] = title_elem.get_text(strip=True) if title_elem else None
            except Exception as e:
                logger.debug(f"Error extracting title: {str(e)}")
                article_data['title'] = None

            # Extract image URL
            try:
                img_elem = article_elem.find('img', src=True)
                article_data['image_url'] = img_elem['src'] if img_elem else None
            except Exception as e:
                logger.debug(f"Error extracting image: {str(e)}")
                article_data['image_url'] = None

            # Extract timestamp
            try:
                parent_div = article_elem.find_parent('div', {'data-timestamp': True})
                article_data['timestamp'] = parent_div['data-timestamp'] if parent_div else None
            except Exception as e:
                logger.debug(f"Error extracting timestamp: {str(e)}")
                article_data['timestamp'] = None

            # Extract views
            try:
                views_elem = article_elem.find('i', class_='icon-eye')
                views = None
                if views_elem and views_elem.find_next('span'):
                    views = views_elem.find_next('span').get_text(strip=True)
                article_data['views'] = views
            except Exception as e:
                logger.debug(f"Error extracting views: {str(e)}")
                article_data['views'] = None

            # Extract category
            try:
                category_elem = article_elem.find('a', class_='post-item-category')
                article_data['category'] = category_elem.get_text(strip=True) if category_elem else None
            except Exception as e:
                logger.debug(f"Error extracting category: {str(e)}")
                article_data['category'] = None

            # Extract likes and dislikes
            try:
                like_btn = article_elem.find('button', class_='like-btn')
                dislike_btn = article_elem.find('button', class_='dislike-btn')

                likes = like_btn.find('span').get_text(strip=True) if like_btn and like_btn.find('span') else None
                dislikes = dislike_btn.find('span').get_text(strip=True) if dislike_btn and dislike_btn.find('span') else None

                article_data['likes'] = likes
                article_data['dislikes'] = dislikes
            except Exception as e:
                logger.debug(f"Error extracting likes/dislikes: {str(e)}")
                article_data['likes'] = None
                article_data['dislikes'] = None

            # Extract tags if present
            try:
                tags = []
                tag_elems = article_elem.find_all('span', class_='post-item-tag')
                for tag in tag_elems:
                    tag_text = tag.get_text(strip=True)
                    if tag_text:
                        tags.append(tag_text)
                article_data['tags'] = tags
            except Exception as e:
                logger.debug(f"Error extracting tags: {str(e)}")
                article_data['tags'] = []

            # Only return if we have at least a title or URL
            if article_data.get('title') or article_data.get('url'):
                return article_data
            else:
                logger.debug("Article has no title or URL, skipping")
                return None

        except Exception as e:
            logger.error(f"Error parsing article: {str(e)}")
            return None

    def parse_page(self, html_content: str, page_number: int) -> List[Dict]:
        """Parse articles from a page with error handling"""
        if not html_content:
            return []

        articles = []

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all article containers
            article_elements = soup.find_all('div', class_='post-item')

            logger.info(f"Found {len(article_elements)} articles on page {page_number}")

            for idx, article_elem in enumerate(article_elements, 1):
                try:
                    article_data = self.parse_article(article_elem)
                    if article_data:
                        article_data['page_number'] = page_number
                        article_data['scraped_at'] = datetime.now().isoformat()
                        articles.append(article_data)
                except Exception as e:
                    logger.error(f"Error parsing article {idx} on page {page_number}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing page {page_number}: {str(e)}")

        return articles

    async def scrape_page(self, session: aiohttp.ClientSession, page_number: int) -> int:
        """Scrape a single page"""
        try:
            html_content = await self.fetch_page(session, page_number)
            if html_content:
                articles = self.parse_page(html_content, page_number)
                self.articles.extend(articles)

                # Save checkpoint after each successful page
                self.save_checkpoint()

                return len(articles)
            return 0
        except Exception as e:
            logger.error(f"Unexpected error scraping page {page_number}: {str(e)}")
            return 0

    async def scrape_all(self):
        """Scrape all pages concurrently with error handling"""
        logger.info(f"Starting scraper from page {self.start_page} to {self.end_page}")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Create session with connection pooling
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=self.timeout)

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                tasks = []
                for page_num in range(self.start_page, self.end_page + 1):
                    task = asyncio.create_task(self.scrape_page(session, page_num))
                    tasks.append(task)

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Count successful results (ignore exceptions)
                total_articles = sum(r for r in results if isinstance(r, int))

                elapsed_time = time.time() - start_time

                logger.info("=" * 60)
                logger.info(f"Scraping complete! Total articles scraped: {total_articles}")
                logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
                logger.info(f"Failed pages: {len(self.failed_pages)}")

                if self.failed_pages:
                    logger.warning("Failed pages:")
                    for failed in self.failed_pages:
                        logger.warning(f"  Page {failed['page']}: {failed['error']}")

        except Exception as e:
            logger.error(f"Critical error during scraping: {str(e)}")
            # Save what we have so far
            self.save_checkpoint()
            raise

    def save_checkpoint(self):
        """Save progress checkpoint"""
        try:
            checkpoint_data = {
                'articles': self.articles,
                'failed_pages': self.failed_pages,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self) -> bool:
        """Load progress from checkpoint"""
        try:
            if Path(self.checkpoint_file).exists():
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    self.articles = checkpoint_data.get('articles', [])
                    self.failed_pages = checkpoint_data.get('failed_pages', [])
                    logger.info(f"Loaded checkpoint with {len(self.articles)} articles")
                    return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
        return False

    def save_to_json(self, filename: str = 'oxu_articles.json'):
        """Save scraped articles to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.articles, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving JSON: {str(e)}")

    def save_to_csv(self, filename: str = 'oxu_articles.csv'):
        """Save scraped articles to CSV file"""
        try:
            import csv

            if not self.articles:
                logger.warning("No articles to save")
                return

            # Get all possible keys
            all_keys = set()
            for article in self.articles:
                all_keys.update(article.keys())

            keys = sorted(all_keys)

            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for article in self.articles:
                    # Convert tags list to string
                    article_copy = article.copy()
                    if 'tags' in article_copy and isinstance(article_copy['tags'], list):
                        article_copy['tags'] = ', '.join(article_copy['tags'])
                    writer.writerow(article_copy)

            logger.info(f"✓ Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")

    def print_summary(self):
        """Print summary statistics"""
        if not self.articles:
            logger.warning("No articles scraped")
            return

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total articles: {len(self.articles)}")

        # Category breakdown
        categories = {}
        for article in self.articles:
            cat = article.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print("\nArticles by category:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")

        # Pages breakdown
        pages = {}
        for article in self.articles:
            page = article.get('page_number', 'Unknown')
            pages[page] = pages.get(page, 0) + 1

        print("\nArticles by page:")
        for page, count in sorted(pages.items()):
            print(f"  Page {page}: {count}")


async def main():
    """Main function to run the scraper"""
    try:
        # Configure scraper
        START_PAGE = 11
        END_PAGE = 15  # Change this to scrape more pages

        scraper = OxuAzScraper(
            start_page=START_PAGE,
            end_page=END_PAGE,
            max_retries=3,
            retry_delay=2,
            request_delay=0.5,  # Delay between requests
            timeout=30
        )

        # Run the scraper
        await scraper.scrape_all()

        # Print summary
        scraper.print_summary()

        # Save results
        scraper.save_to_json('oxu_articles.json')
        scraper.save_to_csv('oxu_articles.csv')

        logger.info("Scraping completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\nScraping interrupted by user")
        logger.info("Saving current progress...")
        scraper.save_to_json('oxu_articles_partial.json')
        scraper.save_to_csv('oxu_articles_partial.csv')
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
