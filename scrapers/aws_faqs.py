import os

from bs4 import BeautifulSoup
from scrapers.base import BaseScraper


class AWSFAQScraper(BaseScraper):
    """
    A class used to scrape AWS FAQs and convert them to PDFs.

    Attributes
    ----------
    base_url : str
        The base URL of the AWS FAQs website.
    output_dir : str, optional
        The directory where the generated PDFs will be saved. Defaults to "docs".

    Methods
    -------
    fetch_page(url)
        Fetches the content of a webpage given its URL.
    normalize_text(input_str)
        Normalizes a string by converting it to lowercase and removing diacritics.
    extract_links(url)
        Extracts all the links to AWS FAQs from a webpage given its URL.
    extract_content(url)
        Extracts the content of an AWS FAQ page given its URL.
    convert_to_pdf(content, filename)
        Converts a string to a PDF and saves it to a file.
    run()
        Runs the scraper by extracting the links to AWS FAQs, extracting their content, and converting it to PDFs.
    """

    def __init__(self, base_url: str, output_dir: str = "docs"):
        self.base_url = base_url
        self.output_dir = output_dir
        # Check if the folder exists if not create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_links(self, url: str) -> list[dict]:
        """
        Extracts all the FAQ question links from the given URL.

        Args:
            url (str): The URL to extract the links from.

        Returns:
            list: A list of dictionaries containing the name and link of each FAQ question.
        """
        links = []
        # Send an HTTP GET request to the URL
        response = self.fetch_page(url)
        if response is None:
            return links

        soup = BeautifulSoup(response, "html.parser")
        # Find all the FAQ question elements
        faq_questions = soup.find_all("div", class_="aws-text-box")
        for question in faq_questions:
            # get the a tag
            a_tag = question.find("a")
            if a_tag:
                # get the href attribute
                href = a_tag.get("href", "")
                # validate if the href has faqs in it
                if "faqs" in href:
                    links.append(
                        {"name": a_tag.text, "link": self.base_url + href})

        return links

    def extract_content(self, url: str) -> str:
        """
        Extracts the content from the given URL by removing specific tags and normalizing the text.

        Args:
            url (str): The URL to extract content from.

        Returns:
            str: The extracted content as a string, or None if the response is empty.
        """
        response = self.fetch_page(url)
        if response is None:
            return None

        soup = BeautifulSoup(response, "html.parser")
        # Remove a specific tag
        breadc = soup.find(
            "div", class_="lb-breadcrumbs lb-breadcrumbs-dropTitle")
        if breadc is not None:
            breadc.clear()

        np = soup.find("div", class_="lb-none-pad lb-grid")
        if np is not None:
            np.clear()

        importants = soup.find_all("div", class_="lb-col lb-tiny-24 lb-mid-24")
        if len(importants) > 0:
            text = "".join(
                [f'<p>{self.normalize_text(important.get_text())}</p>' for important in importants])
            return text

        return None

    def run(self) -> None:
        """
        Extracts links from the base URL and converts the content of each link to a PDF file.

        Returns:
            None
        """
        links = self.extract_links(self.base_url + "/faqs/")
        for link in links:
            content = self.extract_content(link["link"])
            if content is not None:
                self.convert_to_pdf(
                    content, f'{self.output_dir}/{link["name"]}.pdf')
            else:
                print(f'Failed to extract content from {link["name"]}')
