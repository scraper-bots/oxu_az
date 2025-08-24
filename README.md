# Banker.az News Scraper

A Python web scraper for extracting news articles from the Banker.az website. This scraper can extract both news listings from category pages and full article content from individual news pages.

## Features

- **Category Page Scraping**: Extract news listings from category pages with pagination support
- **Article Content Scraping**: Extract full article content, including title, date, category, content, tags, and source
- **Multiple Output Formats**: Save data as JSON, CSV, or custom formats
- **Rate Limiting**: Built-in delays to be respectful to the server
- **Robust Error Handling**: Handles network errors and parsing issues gracefully
- **Logging**: Comprehensive logging for monitoring scraping progress

## Installation

1. Clone or download the files
2. Install required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- `requests` (for HTTP requests)
- `beautifulsoup4` (for HTML parsing)
- `lxml` (for faster XML/HTML parsing)

## Usage

### Basic Usage

```python
from banker_az_scraper import BankerAzScraper

# Initialize the scraper
scraper = BankerAzScraper()

# Scrape a category page
news_list = scraper.scrape_category_page("https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/")

# Scrape an individual article
article_data = scraper.scrape_article("https://banker.az/some-article-url/")

# Save data to JSON
scraper.save_to_json(news_list, 'news_data.json')
```

### Running the Example

The main script includes examples:

```bash
python3 banker_az_scraper.py
```

This will:
1. Scrape the provided category page
2. Extract the provided article content
3. Save results to JSON files
4. Scrape detailed content for the first 3 articles

### Custom Usage Examples

See `example_usage.py` for more detailed usage examples:

```bash
python3 example_usage.py
```

## Data Structure

### News List Item
```json
{
  "title": "Article title",
  "url": "https://banker.az/article-url/",
  "date_iso": "2025-08-21T14:37:49+04:00",
  "date_display": "21/08/2025",
  "image_url": "https://banker.az/wp-content/uploads/image.jpg"
}
```

### Full Article Data
```json
{
  "title": "Article title",
  "url": "https://banker.az/article-url/",
  "date_iso": "2025-08-21T14:37:49+04:00",
  "date_display": "21/08/2025",
  "category": "Category name",
  "image_url": "https://banker.az/wp-content/uploads/image.jpg",
  "content": "Full article text content...",
  "content_html": "<div>Full HTML content...</div>",
  "source": "Source name",
  "tags": ["tag1", "tag2"]
}
```

## Available Methods

### `BankerAzScraper` Class Methods

- `scrape_category_page(url)`: Scrape a single category page
- `scrape_multiple_pages(base_url, max_pages)`: Scrape multiple pages from a category
- `scrape_article(url)`: Scrape an individual article
- `save_to_json(data, filename)`: Save data to JSON file

## Rate Limiting

The scraper includes built-in rate limiting:
- 1 second delay between category page requests
- 2 seconds delay between article requests
- Respects server resources and avoids overwhelming the website

## Error Handling

The scraper includes comprehensive error handling:
- Network timeout handling (30 seconds)
- HTML parsing error recovery
- Missing element graceful handling
- Detailed logging of errors and progress

## Customization

You can customize the scraper by:
- Modifying the `extract_news_list()` method for different HTML structures
- Adjusting rate limiting delays in `scrape_multiple_pages()`
- Adding new extraction methods for additional data fields
- Implementing custom output formats

## Output Files

When running the main script, these files are generated:
- `banker_az_news_list.json`: List of news articles from category page
- `banker_az_article_sample.json`: Sample individual article content
- `banker_az_detailed_articles.json`: Detailed content for first 3 articles

## Compliance

This scraper is designed to be respectful to the website:
- Uses proper User-Agent headers
- Implements rate limiting
- Handles errors gracefully without overwhelming the server
- Only extracts publicly available content

## Troubleshooting

Common issues and solutions:

1. **Import Error**: Make sure all required packages are installed
2. **Network Errors**: Check internet connection and website availability
3. **Parsing Errors**: Website structure may have changed, check HTML selectors
4. **Rate Limiting**: If blocked, increase delays between requests

## License

This scraper is for educational and personal use. Please respect the website's robots.txt and terms of service.