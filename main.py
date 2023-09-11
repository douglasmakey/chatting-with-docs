import argparse
import sys
import runpy

from dotenv import load_dotenv

import chromadb

from langchain.embeddings import HuggingFaceBgeEmbeddings
from app import ChromaDB
from scrapers.aws_faqs import AWSFAQScraper
from scrapers.bg3 import BG3Scraper
from utils import get_device


def scrape(s_args: argparse.Namespace) -> None:
    """
    Execute the specified scrape and saves the results to a file.

    Args:
        args: An argparse.Namespace object containing the following attributes:
            - scrape: A string representing the scrape implementation (e.g. "aws").
            - output_dir: A string representing the directory to save the scraped data to.

    Raises:
        ValueError: If an invalid cloud provider is specified.

    Returns:
        None
    """
    print(f"Scraping {s_args.target}")
    if s_args.target == "aws":
        scraper = AWSFAQScraper("https://aws.amazon.com",
                                output_dir=s_args.output_dir)
        scraper.run()
    elif s_args.target == "bg3":
        scraper = BG3Scraper("https://bg3.wiki/",
                             output_dir=s_args.output_dir)
        scraper.run()
    else:
        raise ValueError("Invalid target")

    print("Done")


def feed(f_args: argparse.Namespace) -> None:
    """
    Feed ChromaDB with data from a directory of PDFs.

    Args:
        args: A Namespace object containing the following attributes:
            - from_path: A string representing the path to the directory containing the PDFs to be fed into ChromaDB.
            - chromadb_persitent_path: A string representing the path to the ChromaDB persistent storage.
            - collection_name: A string representing the name of the collection to store the PDF data in.
    """
    print(f'Feeding ChromaDB with data from {f_args.from_path} directory')
    model_kwargs = {'device':  get_device()}
    encode_kwargs = {'normalize_embeddings': True, "show_progress_bar": True}
    embedding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    client = chromadb.PersistentClient(path=f_args.chromadb_persitent_path)
    vector_db = ChromaDB(embedding_fn=embedding, client=client,
                         collection_name=f_args.collection_name)

    if f_args.data_type not in ["pdf", "txt"]:
        raise ValueError("Invalid data type")

    vector_db.feed_from_path(
        f_args.from_path, split_documents=f_args.split_documents, data_type=f_args.data_type)

    print("Done")


def app(_args: argparse.Namespace) -> None:
    """
    This function sets the command line arguments and runs the Streamlit app.
    """
    sys.argv = ["streamlit", "run", "app.py"]
    runpy.run_module("streamlit", run_name="__main__")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description='Simple GPT based search engine')

    sub_parsers = parser.add_subparsers(dest='command', help='Subcommands')
    sub_parsers.required = True

    # feed
    feed_cmd = sub_parsers.add_parser(
        'feed', help='Feed ChromaDB with documents')

    feed_cmd.add_argument('--from-path', type=str,
                          help='Path to the folder with the documents', required=True)

    feed_cmd.add_argument('--split-documents',
                          action=argparse.BooleanOptionalAction, help='Split documents into chunks')

    feed_cmd.add_argument('--data-type', type=str,
                          default="pdf", choices=['pdf', 'txt'], help='The type of data to feed')

    feed_cmd.add_argument('--collection-name', type=str,
                          help='The name of the collection to create', required=True)

    feed_cmd.add_argument('--chromadb-persitent-path',
                          type=str, default="db", help='The path to the ChromaDB persistent storage')

    feed_cmd.set_defaults(func=feed)

    # scraping
    scraping = sub_parsers.add_parser('scraping', help='execute a scraper')

    scraping.add_argument("--target", type=str, required=True,
                          help='What scrape to choose', choices=['aws', 'bg3'])

    scraping.add_argument('--output-dir', type=str, default="docs")

    scraping.set_defaults(func=scrape)

    # app
    runner = sub_parsers.add_parser('app', help='Run the Streamlit app')
    runner.set_defaults(func=app)

    args = parser.parse_args()
    args.func(args)
