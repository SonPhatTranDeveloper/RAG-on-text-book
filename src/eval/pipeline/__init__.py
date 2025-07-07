from src.eval.llm.openai import OpenAIExtractor
from src.eval.scraper.scrapfly import ScrapflyScraper
from src.eval.pipeline.base import BasePipeline


def create_standard_pipeline() -> BasePipeline:
    """
    Creates a standard pipeline with ScrapflyScraper and OpenAIExtractor.

    Returns:
        BasePipeline: An instance of BasePipeline with the specified scraper and extractor.
    """
    scraper = ScrapflyScraper()
    extractor = OpenAIExtractor()
    return BasePipeline(scraper=scraper, extractor=extractor)
