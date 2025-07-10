from src.eval.llm.openai import OpenAIExtractor
from src.eval.scraper.scrapfly import ScrapflyScraper
from src.eval.ingestion_pipeline.base import BasePipeline, VietJackPipeline


def create_standard_pipeline(pipeline_type: str) -> BasePipeline:
    """
    Creates a standard pipeline with ScrapflyScraper and OpenAIExtractor.

    Returns:
        BasePipeline: An instance of BasePipeline with the specified scraper and extractor.
    """
    scraper = ScrapflyScraper()
    extractor = OpenAIExtractor()

    if pipeline_type == "vietjack":
        return VietJackPipeline(scraper=scraper, extractor=extractor)
    else:
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")
