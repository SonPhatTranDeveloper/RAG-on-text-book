from src.eval.llm.base import BaseExtractor
from src.eval.llm.openai import OpenAIExtractor
from src.eval.scraper.base import BaseScraper
from src.eval.scraper.scrapfly import ScrapflyScraper
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pprint


class BasePipeline:
    def __init__(self, scraper: BaseScraper, extractor: BaseExtractor):
        """
        Initializes the BasePipeline with a scraper and an extractor.

        Args:
            scraper (BaseScraper): An instance of a scraper to fetch HTML content.
            extractor (BaseExtractor): An instance of an extractor to process the HTML content.
        """
        self.scraper = scraper
        self.extractor = extractor

    def run(self, url: str) -> list[dict]:
        """
        Runs the pipeline to scrape HTML content from a URL and extract information.

        Args:
            url (str): The URL to scrape.

        Returns:
            list[dict]: A list of dictionaries containing the extracted question and answer.
        """
        # Get HTML Content and load it into BeautifulSoup
        html_content = self.scraper.scrape(url)
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract only the core content from the HTML
        middle_col = soup.select_one(".content")
        return self.extractor.extract(middle_col.text)


if __name__ == "__main__":
    load_dotenv()
    base_scraper = ScrapflyScraper()
    openai_extractor = OpenAIExtractor()
    pipeline = BasePipeline(scraper=base_scraper, extractor=openai_extractor)
    result = pipeline.run(
        "https://www.vietjack.com/dia-li-10-kn/trac-nghiem-bai-1-mon-dia-li-voi-dinh-huong-nghe-nghiep.jsp"
    )
    pprint.pprint(result)
