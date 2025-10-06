import requests
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from typing import Dict, List, Optional, Any
import logging
import time
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def scrape_url(self, url: str, selectors: Optional[Dict[str, str]] = None,
                   verify_ssl: bool = True, timeout: int = 30,
                   use_selenium: bool = False, wait_for_element: Optional[str] = None,
                   javascript_delay: int = 2) -> Dict[str, Any]:
        """
        Scrape a URL using either requests/BeautifulSoup or Selenium.

        Args:
            url: The URL to scrape
            selectors: CSS selectors to extract specific data
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
            use_selenium: Whether to use Selenium for JavaScript-heavy sites
            wait_for_element: CSS selector to wait for before scraping
            javascript_delay: Delay after page load for JS execution

        Returns:
            Dict containing scraped data
        """
        try:
            if use_selenium:
                return self._scrape_with_selenium(
                    url, selectors, verify_ssl, timeout,
                    wait_for_element, javascript_delay
                )
            else:
                return self._scrape_with_requests(
                    url, selectors, verify_ssl, timeout
                )
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {
                "error": str(e),
                "url": url,
                "success": False
            }

    def _scrape_with_requests(self, url: str, selectors: Optional[Dict[str, str]] = None,
                             verify_ssl: bool = True, timeout: int = 30) -> Dict[str, Any]:
        """Scrape using requests and BeautifulSoup."""
        response = self.session.get(url, verify=verify_ssl, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        result = {
            "url": url,
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type'),
            "title": self._extract_title(soup),
            "text": soup.get_text(strip=True),
            "links": self._extract_links(soup, url),
            "success": True
        }

        # Extract data using selectors
        if selectors:
            result["extracted_data"] = {}
            for key, selector in selectors.items():
                elements = soup.select(selector)
                if elements:
                    if len(elements) == 1:
                        result["extracted_data"][key] = elements[0].get_text(strip=True)
                    else:
                        result["extracted_data"][key] = [
                            elem.get_text(strip=True) for elem in elements
                        ]

        return result

    def _scrape_with_selenium(self, url: str, selectors: Optional[Dict[str, str]] = None,
                             verify_ssl: bool = True, timeout: int = 30,
                             wait_for_element: Optional[str] = None, javascript_delay: int = 2) -> Dict[str, Any]:
        """Scrape using Selenium for JavaScript-heavy sites."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")  # Speed up loading

        if not verify_ssl:
            chrome_options.add_argument("--ignore-certificate-errors")
            chrome_options.add_argument("--ignore-ssl-errors")
            chrome_options.add_argument("--allow-running-insecure-content")

        driver = None
        try:
            driver = webdriver.Chrome(
                ChromeDriverManager().install(),
                options=chrome_options
            )
            driver.get(url)

            # Wait for element if specified
            if wait_for_element:
                WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                )

            # Wait for JavaScript execution
            time.sleep(javascript_delay)

            # Get page source
            soup = BeautifulSoup(driver.page_source, 'lxml')

            result = {
                "url": url,
                "title": self._extract_title(soup),
                "text": soup.get_text(strip=True),
                "links": self._extract_links(soup, url),
                "success": True,
                "used_selenium": True
            }

            # Extract data using selectors
            if selectors:
                result["extracted_data"] = {}
                for key, selector in selectors.items():
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            if len(elements) == 1:
                                result["extracted_data"][key] = elements[0].text
                            else:
                                result["extracted_data"][key] = [
                                    elem.text for elem in elements
                                ]
                    except Exception as e:
                        logger.warning(f"Error extracting {key} with selector {selector}: {str(e)}")
                        result["extracted_data"][key] = None

            return result

        finally:
            if driver:
                driver.quit()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else ""

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(base_url, href)
            links.append({
                "text": a_tag.get_text(strip=True),
                "url": absolute_url,
                "domain": urlparse(absolute_url).netloc
            })
        return links

    def scrape_multiple_urls(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Scrape multiple URLs with the same configuration."""
        results = []
        for url in urls:
            result = self.scrape_url(url, **kwargs)
            results.append(result)
        return results

    def extract_structured_data(self, url: str, schema: Dict[str, Any],
                               verify_ssl: bool = True) -> Dict[str, Any]:
        """
        Extract structured data based on a schema definition.

        Schema format:
        {
            "fields": {
                "title": {"selector": "h1", "attribute": "text"},
                "description": {"selector": ".description", "attribute": "text"},
                "price": {"selector": ".price", "attribute": "text", "regex": r"\$([\d.]+)"}
            },
            "use_selenium": false
        }
        """
        selectors = {}
        for field_name, field_config in schema.get("fields", {}).items():
            selectors[field_name] = field_config["selector"]

        result = self.scrape_url(
            url,
            selectors=selectors,
            verify_ssl=verify_ssl,
            use_selenium=schema.get("use_selenium", False),
            wait_for_element=schema.get("wait_for_element"),
            javascript_delay=schema.get("javascript_delay", 2)
        )

        # Apply additional processing (regex, attribute extraction, etc.)
        if "extracted_data" in result:
            for field_name, field_config in schema.get("fields", {}).items():
                if field_name in result["extracted_data"]:
                    value = result["extracted_data"][field_name]

                    # Apply regex if specified
                    if "regex" in field_config and value:
                        match = re.search(field_config["regex"], str(value))
                        if match:
                            value = match.group(1) if match.groups() else match.group(0)

                    # Extract attribute instead of text
                    if field_config.get("attribute") and field_config["attribute"] != "text":
                        # This would require re-scraping with different logic
                        # For now, we'll keep the text extraction
                        pass

                    result["extracted_data"][field_name] = value

        return result