from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, html_text: str) -> list[dict]:
        """
        Extracts a list of { question, answer } from HTML content.

        Args:
            html_text (str): The input HTML text to extract information from.

        Returns:
            list[dict]: A list of dictionaries containing the extracted question and answer.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
