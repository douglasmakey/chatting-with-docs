import os
from bs4 import BeautifulSoup

from scrapers.base import BaseScraper


class BG3Scraper(BaseScraper):
    """
    A web scraper for the Baldur's Gate 3 database from gamerguides.
    Website: https://www.gamerguides.com/baldurs-gate-3/database

    Attributes:
        base_url (str): The base URL of the Baldur's Gate 3 database.
        output_dir (str): The directory where the scraped data will be saved.
    """

    def __init__(self, base_url: str, output_dir: str = "docs"):
        self.base_url = base_url
        self.output_dir = output_dir
        # Check if the folder exists if not create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_item_links(self, content) -> list[str]:
        """
        Extracts item links from the given HTML content.

        Args:
            content (str): The HTML content to extract item links from.

        Returns:
            list[str]: A list of item links extracted from the HTML content.
        """
        links = set()
        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("tbody")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                name = row.find("td")
                if name is None:
                    continue
                link = name.find("a")
                if link is None:
                    continue
                links.add(link["href"])

        return links

    def get_all_items(self):
        """
        Fetches all items from the base_url and saves them as PDF files in the output_dir.

        Returns:
            None
        """
        urls = [
            "wiki/Clothing",
            "wiki/Armor",
            "wiki/Shields",
            "wiki/Headwear",
            "wiki/Cloaks",
            "wiki/Handwear",
            "wiki/Footwear",
            "wiki/Amulets",
            "wiki/Rings",
            "wiki/Arrows",
            "wiki/List_of_Weapons"
        ]

        for url in urls:
            content = self.fetch_page(self.base_url + url)
            links = self.extract_item_links(content)
            for link in links:
                print(f"Fetching {link}")
                content = self.fetch_page(self.base_url + link)
                if content is None:
                    continue

                soup = BeautifulSoup(content, "html.parser")
                body = soup.find("div", class_="mw-parser-output")
                if body is None:
                    continue

                edit_secs = body.find_all("span", class_="mw-editsection")
                for edit in edit_secs:
                    edit.decompose()

                body_text = body.get_text()

                # Get image
                image_div = body.find("div", class_="floatright")
                if image_div is not None:
                    image = image_div.find("img")
                    # append image to body text
                    body_text += f"\n\n image: {self.base_url+image['src']}"

                self.convert_to_pdf(
                    self.normalize_text(body_text),
                    f"{self.output_dir}/{link.split('/')[-1]}.pdf",
                )

    def extract_spells_links(self):
        """
        Extracts links to all spells from the Baldur's Gate 3 wiki.

        Returns:
            A set of links to all spells on the wiki.
        """
        result = set()
        content = self.fetch_page(f"{self.base_url}wiki/Spells#All_Spells")
        soup = BeautifulSoup(content, "html.parser")
        divs = soup.find_all("div", class_="div-col")
        for div in divs:
            links = div.find_all("a")
            for link in links:
                result.add(link["href"])
        return result

    def get_spells(self):
        """
        Extracts spells links, fetches the content of each link, extracts the spell's body text, 
        and converts it to a PDF file. If an image is found, it is appended to the body text before 
        converting it to PDF. If variants are found, their links are added to the list of links to extract.
        """
        links = self.extract_spells_links()
        i = 0
        while i < len(links):
            link = links.pop()
            print(f"Fetching {self.base_url + link}")
            content = self.fetch_page(self.base_url + link)
            if content is None:
                continue

            soup = BeautifulSoup(content, "html.parser")
            body = soup.find("div", class_="mw-parser-output")
            if body is None:
                continue

            edit_secs = body.find_all("span", class_="mw-editsection")
            for edit in edit_secs:
                edit.decompose()

            # Get variants
            variants_title = body.find("span", id="Variants")
            if variants_title is not None:
                variant_parent = variants_title.parent
                if variant_parent is not None:
                    variants = variant_parent.find_next("ul")
                    for variant in variants:
                        variant_link = variant.find("a")
                        if variant_link is not None and variant_link != -1:
                            links.add(variant_link["href"])

            body_text = body.get_text()
            # Get image
            image_div = body.find("div", class_="floatright")
            if image_div is not None:
                image = image_div.find("img")
                # append image to body text
                body_text += f"\n\n image: {self.base_url+image['src']}"

            self.convert_to_pdf(
                self.normalize_text(body_text),
                f"{self.output_dir}/{link.split('/')[-1]}.pdf",
            )
            i += 1

    def get_feats(self):
        """
        Fetches the feats from the base_url's wiki page and converts them to PDFs.

        Returns:
            None
        """
        content = self.fetch_page(f"{self.base_url}wiki/Feats")
        soup = BeautifulSoup(content, "html.parser")
        body = soup.find("div", class_="mw-parser-output")
        if body is None:
            return

        edit_secs = body.find_all("span", class_="mw-editsection")
        for edit in edit_secs:
            edit.decompose()

        table = body.find("table", class_="wikitable")
        if table is None:
            return

        rows = table.find_all("tr")
        for row in rows[1:]:
            name = row.find("td").get_text()
            description = row.find("td").find_next("td").get_text()
            self.convert_to_pdf(
                self.normalize_text(f"{name}: {description}\n\n"),
                f"{self.output_dir}/feats-{name}.pdf",
            )

    def extract_locations_links(self):
        """
        Extracts links to all locations from the List of Locations page on the Baldur's Gate 3 wiki.

        Returns:
            A set of links to all locations on the Baldur's Gate 3 wiki.
        """
        links = set()
        content = self.fetch_page(f"{self.base_url}wiki/List_of_Locations")
        soup = BeautifulSoup(content, "html.parser")
        body = soup.find("div", class_="mw-parser-output")
        uls = body.find_all("ul")
        for ul in uls:
            lis = ul.find_all("li")
            for li in lis:
                link = li.find("a")
                if link is not None:
                    links.add(link["href"])

        return links

    def get_locations(self):
        """
        Extracts the locations links, fetches the content of each location, removes edit sections from the HTML, 
        converts the body text to PDF and saves it to the output directory.
        """
        locations = self.extract_locations_links()
        for location in locations:
            print(f"Fetching {self.base_url + location}")
            content = self.fetch_page(self.base_url + location)
            if content is None:
                continue

            soup = BeautifulSoup(content, "html.parser")
            body = soup.find("div", class_="mw-parser-output")
            if body is None:
                continue

            edit_secs = body.find_all("span", class_="mw-editsection")
            for edit in edit_secs:
                edit.decompose()

            body_text = body.get_text()
            self.convert_to_pdf(
                self.normalize_text(body_text),
                f"{self.output_dir}/{location.split('/')[-1]}.pdf",
            )

    def run(self):
        """
        Runs the scraper to get all items, spells, feats, and locations.
        """
        self.get_all_items()
        self.get_spells()
        self.get_feats()
        self.get_locations()
