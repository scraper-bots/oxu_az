#!/usr/bin/env python3
"""Quick test of async scraper"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_all_banker_news import BankerAzExtractor

async def quick_test():
    """Quick test of async scraper functionality."""
    print("🧪 Testing Async Banker.az Scraper")
    print("=" * 40)
    
    async with BankerAzExtractor(concurrent_requests=5, delay=0.5) as extractor:
        # Test 1: Scrape single category page
        print("Test 1: Single category page...")
        category_url = "https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/"
        news_items = await extractor.scrape_category_pages(category_url, max_pages=2)
        print(f"✅ Found {len(news_items)} news items")
        
        # Show sample items
        for i, item in enumerate(news_items[:3]):
            print(f"  {i+1}. {item['title'][:50]}...")
        
        # Test 2: Scrape article content
        if news_items:
            print(f"\nTest 2: Article content extraction...")
            sample_url = news_items[0]['url']
            article_content = await extractor.scrape_articles_content([sample_url])
            
            if article_content and article_content[0]:
                article = article_content[0]
                print(f"✅ Article: {article.get('title', 'N/A')[:50]}...")
                print(f"   Content length: {len(article.get('content', ''))}")
                print(f"   Word count: {article.get('word_count', 0)}")
                print(f"   Category: {article.get('category', 'N/A')}")
                print(f"   Tags: {article.get('tags', [])}")
            else:
                print("❌ Failed to extract article content")
        
        # Test 3: Save data
        print(f"\nTest 3: Data saving...")
        extractor.save_data(news_items[:5], "test_async_output")
        print("✅ Data saved successfully")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    try:
        asyncio.run(quick_test())
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()