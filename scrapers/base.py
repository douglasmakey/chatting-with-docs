import html
import unicodedata
import requests
import pdfkit

from pathlib import Path


class BaseScraper:
    """
    Base class for web scrapers.

    This class provides methods for fetching the content of a webpage, normalizing text, writing content to a file, and converting content to a PDF file.
    """

    def fetch_page(self, url: str) -> str:
        """
        Fetches the content of a webpage at the given URL.

        Args:
            url (str): The URL of the webpage to fetch.

        Returns:
            str: The content of the webpage, if the request was successful. None otherwise.
        """
        header = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        }

        try:
            with requests.Session() as session:
                response = session.get(url, timeout=10, headers=header)
                if response.status_code == 200:
                    return response.text
                else:
                    print(
                        f'Failed to retrieve the webpage: {url} status code: {response.status_code}')
                    return None
        except requests.RequestException as error:
            print(f'Error occurred during requests to {url} : {error}')
            return None

    def normalize_text(self, input_str: str) -> str:
        """
        Normalizes the input string by converting it to lowercase, unescaping any HTML entities, and removing any diacritical marks.

        Args:
            input_str (str): The string to be normalized.

        Returns:
            str: The normalized string.
        """
        input_str = input_str.lower()
        input_str = html.unescape(input_str)
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        normalized_str = ''.join(
            [c for c in nfkd_form if not unicodedata.combining(c)])

        # replace tabs and newlines with spaces
        normalized_str = normalized_str.replace('\n', ' ').replace('\t', ' ')
        return normalized_str

    def write_to_file(self, content: str, filename: str) -> None:
        """
        Writes the given content to a file with the given filename.

        Args:
            content (str): The content to write to the file.
            filename (str): The name of the file to write the content to.

        Returns:
            None
        """
        file_path = Path(filename + ".txt")
        try:
            with file_path.open("w", encoding="utf-8") as file:
                file.write(content)
        except FileNotFoundError:
            print(f"Error: Directory '{file_path.parent}' does not exist.")

    def convert_to_pdf(self, content: str, filename: str) -> None:
        """
        Converts the given content to a PDF file and saves it with the given filename.

        Args:
            content (str): The content to convert to PDF.
            filename (str): The name of the file to save the PDF as.

        Returns:
            None
        """

        options = {
            '--no-print-media-type': ''
        }

        pdfkit.from_string(content, filename, options=options)
