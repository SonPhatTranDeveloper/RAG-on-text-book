from abc import ABC, abstractmethod


class BaseScraper(ABC):
    @abstractmethod
    def scrape(self, url: str) -> str:
        """Scrape a website and return content

        Args:
            url (str): URL of the content

        Raises:
            NotImplementedError: Not implemented in the base class.

        Returns:
            str: HTML content of the page.
        """
        raise NotImplementedError("This method should be implemeted by child classes.")
