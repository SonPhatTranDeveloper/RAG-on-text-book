from scrapfly import ScrapflyClient, ScrapeConfig, ScrapeApiResponse
import os
from src.eval.scraper.base import BaseScraper
from typing import Optional


class ScrapflyScraper(BaseScraper):
    def __init__(self, api_key: Optional[str] = None):
        """Initializes the ScrapflyScraper with an API key.
        Args:
            api_key (str): Your Scrapfly API key. If not provided, it will attempt to read from the SCRAPFLY_API_KEY environment variable.
        """
        if not api_key:
            api_key = os.getenv("SCRAPFLY_API_KEY")
        self.client = ScrapflyClient(key=api_key)

    def scrape(self, url: str) -> str:
        """Scrape a website and return content.

        Args:
            url (str): URL of the content

        Returns:
            str: HTML content of the page.
        """
        result: ScrapeApiResponse = self.client.scrape(
            ScrapeConfig(
                tags=["player", "project:default"],
                proxy_pool="public_residential_pool",
                asp=True,
                cost_budget=200,
                url=url,
            )
        )
        return result.scrape_result["content"]
