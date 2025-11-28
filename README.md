# Oxu.az Web Scraper

Asynchronous web scraper for oxu.az news website using Python's asyncio and aiohttp.

## Features

- **Async scraping** with aiohttp for high performance
- **Concurrent requests** to scrape multiple pages simultaneously
- **Comprehensive data extraction**:
  - Article title
  - Article URL
  - Image URL
  - Timestamp
  - Views count
  - Category
  - Likes/dislikes
  - Tags (photo, video, etc.)
- **Multiple output formats**: JSON and CSV
- **Summary statistics** by category and page
- **Error handling** for robust scraping

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the scraper with default settings (pages 11-15):
```bash
python scraper.py
```

### Custom Configuration

Edit the `main()` function in `scraper.py` to customize:

```python
# Scrape pages 11 to 20
START_PAGE = 11
END_PAGE = 20

scraper = OxuAzScraper(start_page=START_PAGE, end_page=END_PAGE)
```

### Programmatic Usage

```python
import asyncio
from scraper import OxuAzScraper

async def custom_scrape():
    # Create scraper instance
    scraper = OxuAzScraper(start_page=1, end_page=50)

    # Run scraper
    await scraper.scrape_all()

    # Save results
    scraper.save_to_json('my_articles.json')
    scraper.save_to_csv('my_articles.csv')

    # Print summary
    scraper.print_summary()

# Run
asyncio.run(custom_scrape())
```

## Output Files

The scraper generates two output files:

1. **oxu_articles.json** - JSON format with full article data
2. **oxu_articles.csv** - CSV format for Excel/spreadsheet analysis

### CSV Columns

- `url` - Article URL
- `title` - Article title
- `image_url` - Article image URL
- `timestamp` - Publication timestamp
- `views` - View count
- `category` - Article category
- `likes` - Number of likes
- `dislikes` - Number of dislikes
- `tags` - Article tags (comma-separated)
- `page_number` - Source page number
- `scraped_at` - When the article was scraped

## Example Output

```
Starting scraper from page 11 to 15
============================================================
✓ Successfully fetched page 11
✓ Successfully fetched page 12
✓ Successfully fetched page 13
Found 13 articles on page 11
Found 13 articles on page 12
Found 13 articles on page 13
============================================================
Scraping complete! Total articles scraped: 65

============================================================
SUMMARY STATISTICS
============================================================
Total articles: 65

Articles by category:
  Cəmiyyət: 20
  İdman: 15
  Dünya: 12
  Siyasət: 10
  Hadisə: 8

Articles by page:
  Page 11: 13
  Page 12: 13
  Page 13: 13
  Page 14: 13
  Page 15: 13

✓ Data saved to oxu_articles.json
✓ Data saved to oxu_articles.csv
```

## Performance

- Scrapes pages concurrently using asyncio
- Typical speed: ~1-2 seconds per page (depending on network)
- Example: Scraping 10 pages takes ~5-10 seconds total

## Error Handling

- Handles network errors gracefully
- Continues scraping even if some pages fail
- Logs errors to console for debugging

## Notes

- Respects website structure and doesn't overwhelm servers
- Uses standard User-Agent header
- Consider adding delays between requests for very large scraping jobs
