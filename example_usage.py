#!/usr/bin/env python3
"""
Example usage of Banker.az scraper
"""

from banker_az_scraper import BankerAzScraper
import json

def main():
    # Initialize the scraper
    scraper = BankerAzScraper()
    
    # Example 1: Scrape a specific category page
    print("=== Example 1: Scraping a category page ===")
    category_url = "https://banker.az/category/x%c9%99b%c9%99rl%c9%99r/"
    news_list = scraper.scrape_category_page(category_url)
    print(f"Found {len(news_list)} articles")
    
    # Print first few titles
    for i, article in enumerate(news_list[:5]):
        print(f"{i+1}. {article['title']}")
        print(f"   Date: {article['date_display']}")
        print(f"   URL: {article['url']}")
        print()
    
    # Example 2: Scrape multiple pages from a category
    print("\n=== Example 2: Scraping multiple pages ===")
    all_news = scraper.scrape_multiple_pages(category_url, max_pages=3)
    print(f"Found {len(all_news)} articles across multiple pages")
    
    # Example 3: Scrape full article content
    if news_list:
        print("\n=== Example 3: Scraping full article content ===")
        first_article_url = news_list[0]['url']
        article_data = scraper.scrape_article(first_article_url)
        
        print(f"Title: {article_data.get('title', 'N/A')}")
        print(f"Category: {article_data.get('category', 'N/A')}")
        print(f"Date: {article_data.get('date_display', 'N/A')}")
        print(f"Source: {article_data.get('source', 'N/A')}")
        
        content = article_data.get('content', '')
        if content:
            print(f"Content preview: {content[:200]}...")
        
        tags = article_data.get('tags', [])
        if tags:
            print(f"Tags: {', '.join(tags)}")
    
    # Example 4: Save data to different formats
    print("\n=== Example 4: Saving data ===")
    
    # Save as JSON
    scraper.save_to_json(news_list, 'example_news_list.json')
    print("Data saved to example_news_list.json")
    
    # You can also save as CSV or other formats
    import csv
    with open('example_news_list.csv', 'w', newline='', encoding='utf-8') as csvfile:
        if news_list:
            fieldnames = ['title', 'url', 'date_display', 'date_iso', 'image_url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in news_list:
                writer.writerow({k: article.get(k, '') for k in fieldnames})
    print("Data saved to example_news_list.csv")

if __name__ == "__main__":
    main()