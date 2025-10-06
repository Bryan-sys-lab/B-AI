#!/usr/bin/env python3
"""
Simple test script for the web scraper agent.
"""

import asyncio
import httpx
import json

async def test_web_scraper():
    """Test the web scraper agent endpoints."""

    base_url = "http://localhost:8012"

    # Test health endpoint
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Test scraping a simple URL
    test_url = "https://httpbin.org/html"
    selectors = {
        "title": "h1",
        "body": "body"
    }

    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "url": test_url,
                "selectors": selectors,
                "verify_ssl": True,
                "timeout": 30
            }
            response = await client.post(f"{base_url}/scrape", json=payload)
            print(f"Scrape response: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Scraped data: {json.dumps(result, indent=2)}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Scrape test failed: {e}")

    # Test execute endpoint
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "description": f"Scrape the title from {test_url}"
            }
            response = await client.post(f"{base_url}/execute", json=payload)
            print(f"Execute response: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Execute result: {json.dumps(result, indent=2)}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Execute test failed: {e}")

if __name__ == "__main__":
    print("Testing web scraper agent...")
    asyncio.run(test_web_scraper())