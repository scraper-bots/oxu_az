# Complete Banker.az News Scraper

**High-performance async scraper** for extracting complete news data from Banker.az with full content extraction, advanced cleaning, and comprehensive metadata.

## 🚀 Features

- **🔥 Async/Aiohttp Optimization**: Up to 10x faster than sync scrapers
- **📰 Complete Content Extraction**: Full article text, not just summaries  
- **🧹 Advanced Cleaning**: Removes ads, social media, promotional content
- **📊 Rich Metadata**: Categories, tags, sources, reading time, word count
- **📁 Single Output File**: One comprehensive JSON file with all data
- **🛡️ Error Handling**: Retry logic, timeout handling, graceful failures
- **⚡ Smart Rate Limiting**: Server-friendly with configurable delays

## 📦 Installation

```bash
# Install required packages
pip install aiohttp beautifulsoup4 lxml

# Or use requirements file
pip install -r requirements.txt
```

## 🎯 Usage

### Simple Extraction

```bash
# Run the complete extractor
python3 complete_banker_extractor.py
```

This will:
- Extract news from multiple categories
- Get complete article content 
- Clean all promotional/ad content
- Save everything to a single JSON file
- Include extraction statistics

### Programmatic Usage

```python
import asyncio
from complete_banker_extractor import CompleteBankerExtractor, extract_all_banker_news_complete

# Simple usage
async def main():
    async with CompleteBankerExtractor() as extractor:
        # Get news list from a category
        news_list = await extractor.scrape_category_pages(
            "https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/", 
            max_pages=5
        )
        
        # Get complete article content
        articles = await extractor.scrape_articles_complete([item['url'] for item in news_list[:10]])
        
        # Save data to single JSON file
        extractor.save_single_json({"articles": articles}, "my_extraction.json")

# Or use the complete extraction function
result = await extract_all_banker_news_complete()

asyncio.run(main())
```

## 📊 Data Structure

### Single JSON File Structure
```json
{
  "extraction_info": {
    "extraction_timestamp": "2025-08-24T22:53:42.503165",
    "total_news_items": 90,
    "complete_articles": 50,
    "categories_found": ["Təhsil", "Vergi xəbərləri", "Energetika", ...],
    "sources_found": ["MODERN.AZ", "REPORT", "APA", ...],
    "content_stats": {
      "avg_word_count": 275,
      "total_words": 13741,
      "avg_reading_time": 1.56
    }
  },
  "news_list": [
    {
      "title": "Article title",
      "url": "https://banker.az/article-url/",
      "category": "Təhsil",
      "date_display": "24/08/2025",
      "image_url": "image.jpg"
    }
  ],
  "articles_with_content": [
    {
      "title": "Article title in Azerbaijani",
      "content": "Complete clean article content...",
      "word_count": 1170,
      "reading_time_minutes": 6,
      "tags": ["əmək bazarı", "ixtisas seçimi"],
      "source": "MODERN.AZ",
      "category": "Təhsil",
      ...
    }
  ]
}
```

## 🎛️ Configuration

### Default Categories
- **News/Reports**: `https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/`
- **Economy**: `https://banker.az/category/iqtisadiyyat/`  
- **Government**: `https://banker.az/category/dovl%c9%99t/`
- **Politics**: `https://banker.az/category/siyaset/`

### Performance Settings
- **Concurrent Requests**: 6-8 simultaneous requests
- **Rate Limiting**: 0.5 seconds between requests
- **Page Processing**: 8 pages processed simultaneously
- **Article Processing**: 10 articles processed simultaneously  
- **Retry Logic**: 3 attempts with exponential backoff
- **Maximum Scraping**: No page limits - extracts ALL available content

## 📁 Output File

After running the scraper, you'll get a **single comprehensive JSON file**:

**`banker_az_complete_data_TIMESTAMP.json`** - Contains everything:
- `extraction_info` - Statistics and metadata about the extraction
- `news_list` - All news articles from category pages (90+ items)
- `articles_with_content` - Complete article content with full text (50+ items)

## 🧹 Content Cleaning

The scraper removes:
- ✅ Social media sharing buttons
- ✅ WhatsApp/Telegram promotional links
- ✅ Advertisement blocks
- ✅ Navigation elements
- ✅ Promotional content (⚡️📲 patterns)
- ✅ Empty or very short paragraphs

## 📈 Performance

**Maximum Extraction Results:**
- **ALL available news articles** from each category (no limits)
- **Complete content** extracted for ALL found articles
- **Automatic stopping** when no more pages are available
- **Hundreds to thousands** of articles depending on category size
- **All categories** identified automatically
- **All sources** and metadata extracted

## 🔧 Advanced Usage

### Custom Category Extraction
```python
categories = [
    "https://banker.az/category/custom-category/",
    "https://banker.az/category/another-category/"
]

result = await extract_all_banker_news_complete(
    categories=categories,
    max_pages_per_category=10,
    max_articles=100
)
```

### Error Handling
The scraper includes comprehensive error handling:
- Network timeouts and connection errors
- HTML parsing failures  
- Missing content graceful degradation
- Detailed logging for debugging

## 🛡️ Best Practices

- **Respectful Scraping**: Built-in rate limiting and retry logic
- **Clean Content**: Advanced text processing for Azerbaijani content
- **Data Quality**: Multiple fallback selectors for robust extraction
- **Performance**: Async processing with smart concurrency control

## 📋 Requirements

- Python 3.7+
- aiohttp >= 3.8.0
- beautifulsoup4 >= 4.9.3  
- lxml >= 4.6.3

## 🎯 Use Cases

Perfect for:
- 📈 News analysis and research
- 📊 Content aggregation
- 🔍 Market research  
- 📰 Media monitoring
- 📚 Dataset creation
- 🤖 NLP/ML training data

## ⚠️ Compliance

This scraper:
- Respects server resources with rate limiting
- Only extracts publicly available content
- Uses proper HTTP headers and practices
- Follows ethical scraping guidelines

## 📞 Support

For issues or questions, check the generated log files and extraction summaries for detailed debugging information.