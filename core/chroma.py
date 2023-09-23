from typing import List

import chromadb
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFium2Loader, UnstructuredMarkdownLoader
from langchain.embeddings.base import Embeddings
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class ChromaDB:
    """
    A class representing a wrapper for Chroma database.

    Attributes:
        embedding (Embeddings): The embedding function used to encode text into vectors.
        vectordb (Chroma): The Chroma vector database.

    Methods:
        get_retriever(k: int = 5) -> BaseRetriever: Returns a retriever object that can be used to search the vector database.
        load_pdfs(path: str) -> List[Document]: Load PDF documents from a given directory path.
        load_text(path: str) -> List[Document]: Load text documents from a given directory path.
        split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]: Splits a list of documents into smaller chunks of text.
        feed_from_pdfs(path: str, chunk_size: int = 600, chunk_overlap: int = 150) -> None: Loads PDFs from a given directory path, splits them into chunks of text, and feeds them into a Chroma database.
    """

    def __init__(self, embedding_fn: Embeddings, client: chromadb.API, collection_name: str):
        self.embedding = embedding_fn
        self.client = client
        self.collection_name = collection_name
        self.vectordb = Chroma(
            embedding_function=embedding_fn, client=client, collection_name=collection_name)

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """
        Returns a retriever object that can be used to search the vector database.

        Args:
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            VectorDBRetriever: A retriever object that can be used to search the vector database.
        """
        return self.vectordb.as_retriever(search_kwargs={"k": k})

    def load_pdfs(self, path: str) -> List[Document]:
        """
        Load PDF documents from a given directory path.

        :param path: The directory path to load PDF documents from.
        :type path: str
        :return: A list of loaded PDF documents.
        :rtype: list
        """
        loader = DirectoryLoader(path, glob="./*.pdf",
                                 loader_cls=PyPDFium2Loader)
        documents = loader.load()
        return documents

    def load_texts(self, path: str) -> List[Document]:
        """
        Load text documents from a given directory path.

        Args:
            path (str): The path to the directory containing the text documents.

        Returns:
            list: A list of loaded text documents.
        """
        loader = DirectoryLoader(path, glob="./*.txt", loader_cls=TextLoader)
        documents = loader.load()
        return documents

    def load_mds(self, path: str) -> List[Document]:
        """
        Load text documents from a given directory path.

        Args:
            path (str): The path to the directory containing the text documents.

        Returns:
            list: A list of loaded text documents.
        """
        loader = DirectoryLoader(path, glob="./*.md",
                                 loader_cls=UnstructuredMarkdownLoader)
        documents = loader.load()
        return documents

    def split_documents(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """
        Splits a list of documents into smaller chunks of text.

        Args:
            documents (list): A list of documents to split.
            chunk_size (int): The maximum size of each chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.

        Returns:
            list: A list of smaller text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts

    def feed_from_path(self, path: str, split_documents: bool = True, chunk_size: int = 600, chunk_overlap: int = 150, data_type: str = "pdf") -> None:
        """
        Loads documents from a given path and feeds them into a Chroma collection.

        Args:
            path (str): The path to the directory containing the documents to be loaded.
            collection_name (str): The name of the Chroma collection to feed the documents into.
            chunk_size (int, optional): The size of each document chunk in characters. Defaults to 600.
            chunk_overlap (int, optional): The number of characters to overlap between document chunks. Defaults to 150.
            data_type (str, optional): The type of data to load. Must be either "pdf" or "txt". Defaults to "pdf".

        Raises:
            ValueError: If an invalid data type is provided.

        Returns:
            None
        """

        if data_type == "pdf":
            docs = self.load_pdfs(path)
        elif data_type == "txt":
            docs = self.load_texts(path)
        else:
            raise ValueError("Invalid data type")

        if split_documents:
            docs = self.split_documents(docs, chunk_size, chunk_overlap)

        Chroma.from_documents(documents=docs, embedding=self.embedding,
                              client=self.client, collection_name=self.collection_name)
