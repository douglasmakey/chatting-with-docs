import os
import textwrap
import yaml
import streamlit as st
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings

from chroma import ChromaDB
from chromadb import chromadb

from utils import get_device, save_files_to_disk


class App:
    """
    A class representing a question-answering app for cloud provider FAQs usign streamlit as UI.

    Attributes:
    - llm (LanguageModel): The language model used for generating answers.
    - qa (RetrievalQA): The question-answering model used for retrieving answers.
    - retriever (Retriever): The retriever used for retrieving relevant documents.
    """

    def __init__(self, embedding: Embeddings, cli: chromadb.API):
        self.default_model_name = "gpt-3.5-turbo-16k"
        self.chromadb_cli = cli
        self.embedding = embedding
        self.llm: ChatOpenAI = ChatOpenAI(model_name=self.default_model_name)
        # Read YAML file
        self.read_config()

        self.k = self.config.get("k", 5)
        self.collection_selected = None
        self.prompt_template_name = "default"
        self.prompt_template = None
        self.retrievel_qa = None

    def read_config(self) -> None:
        """
        Reads the configuration file and loads it into the `config` attribute of the object.

        :return: None
        """
        filename = "config.yaml"
        with open(filename, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

    def instance_chain(self) -> None:
        """
        Initializes a RetrievalQA instance with the specified parameters and sets it as the `retrieval_qa` attribute of the object.

        If `self.collection_selected` is None, the method returns without doing anything.

        Prints information about the model name, collection, and prompt template being used.

        If `self.prompt_template` is None, the `chain_type_kwargs` dictionary is empty. Otherwise, it contains a "prompt" key with the value of `self.prompt_template`.

        The `RetrievalQA` instance is created using the `from_chain_type` method with the following parameters:
        - `llm`: the `LanguageModel` instance to use
        - `chain_type`: the type of chain to use (in this case, "stuff")
        - `retriever`: the `Retriever` instance to use
        - `chain_type_kwargs`: the dictionary of additional keyword arguments to pass to the chain type constructor
        - `return_source_documents`: whether to return the source documents along with the answers

        The resulting `RetrievalQA` instance is stored as the `retrieval_qa` attribute of the object.
        """

        if self.collection_selected is None:
            return

        print(
            f"Instance OpenAI model: {self.llm.model_name} with the collection: {self.collection_selected} and prompt: {self.prompt_template_name}")

        if self.prompt_template is None:
            chain_type_kwargs = {}
        else:
            chain_type_kwargs = {"prompt": self.prompt_template}

        self.retrievel_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=ChromaDB(self.embedding, self.chromadb_cli,
                               self.collection_selected).get_retriever(k=self.k),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

    def sidebar(self) -> None:
        """
        Renders the sidebar of the SimpleGPT app, which includes fields for entering an OpenAI API key, selecting a
        collection of documents, choosing a prompt, and deleting a collection. Also displays instructions for how to use
        the app and information about SimpleGPT.
        """
        with st.sidebar:
            st.markdown(
                "## How to use\n"
                "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
                "2. Choose your collection of documents 📄\n"
                "3. Ask a question about your collection\n"
            )

            key = "OPENAI_API_KEY"
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Paste your OpenAI API key here (sk-...)",
                help="Get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
                value=os.environ.get(
                    key, None) or st.session_state.get(key, ""),
            )

            model_option = st.text_input(
                'Model name', value=self.default_model_name)

            if model_option:
                self.llm = ChatOpenAI(model_name=model_option)

            available_collections = []
            for collection in self.chromadb_cli.list_collections():
                available_collections.append(collection.name)

            collection_option = st.selectbox(
                'Choose your collection', available_collections, key="collection_option")

            self.collection_selected = collection_option
            if collection_option:
                if st.button("Delete collection", type="primary"):
                    self.chromadb_cli.delete_collection(collection_option)
                    st.success(
                        f"Collection ***{collection_option}*** deleted successfully.")
                    st.experimental_rerun()

            available_prompts = ["default"]
            for prompt in self.config["prompts"]:
                available_prompts.append(prompt["name"])

            prompt_option = st.selectbox(
                'Choose your prompt', available_prompts)

            for prompt in self.config["prompts"]:
                if prompt["name"] == prompt_option:
                    self.prompt_template_name = prompt["name"]
                    self.prompt_template = PromptTemplate(
                        template=prompt["template"], input_variables=["context", "question"])

            with st.spinner("Loading parameters..."):
                self.instance_chain()

            st.session_state[key] = openai_key
            st.markdown("---")
            st.markdown("# About")
            st.markdown(
                "SimpleGPT allows you to ask questions about your "
                "documents and get accurate answers while you can play with the prompt to get better results."
                "\n\n SimpleGPT: Use OpenAI's chat GPT models and BAAI/bge-base-en as embedding model."
            )

    def wrap_text_preserve_newlines(self, text: str, width: int = 110) -> str:
        """
        Wraps the input text to the specified width while preserving any existing newline characters.

        Args:
            text (str): The text to wrap.
            width (int, optional): The maximum width of each line. Defaults to 110.

        Returns:
            str: The wrapped text.
        """
        # Split the input text into lines based on newline characters
        lines = text.split('\n')
        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text

    def generate_html_response(self, query: str, response: dict) -> str:
        """
        Generates an HTML response for the given query and response.

        Args:
            query (str): The query that was asked.
            response (dict): The response to the query.

        Returns:
            str: The HTML response.
        """
        text = f'<p><span style="font-weight: bold;">Q:</span> {query}</p>'
        wrapper_text = self.wrap_text_preserve_newlines(response["result"])
        text += f'<p><span style="font-weight: bold;">A:</span>{wrapper_text}</p>'

        # Add Sources to the text
        text += '<p style="font-weight: bold;">Sources:</p>'
        text += "<ul>"

        for source in response["source_documents"]:
            text += f'<li style="list-style-type: none;">{source.metadata["source"]}</li>'
        text += "</ul>"

        # Wrap the HTML content in a div with a chat-response class
        html = f'<div style="background-color: rgb(38, 39, 48); padding: 10px; border-radius: 5px;">{text}</div>'

        return html

    def qa_tab(self) -> None:
        """
        Displays the QA tab in the Streamlit app. If the OpenAI API key is not set, a warning message is displayed.
        If a prompt template is selected, it is displayed in an expander. If a collection is not selected, a warning message
        is displayed. If a user enters a question, the method retrieves the answer from the selected collection and displays
        it in an HTML format.
        """
        openai_api_key = st.session_state.get("OPENAI_API_KEY")
        if not openai_api_key:
            st.warning(
                "Enter your OpenAI API key in the sidebar. You can get a key at"
                " https://platform.openai.com/account/api-keys."
            )

        if self.prompt_template:
            with st.expander("See the selected prompt"):
                st.markdown(
                    ":orange[If you want to change or create a prompt go to the config.yaml file.]")
                st.divider()
                st.write(self.prompt_template.template)

        if not self.collection_selected:
            st.warning(
                "Choose a collection in the sidebar or crate a new one."
            )

        if self.retrievel_qa is not None:
            user_input = st.text_input(
                f"Ask a question about your collection: ***{self.collection_selected}***")

            if user_input:
                with st.spinner("Thinking..."):
                    response = self.retrievel_qa(user_input)
                    html_response = self.generate_html_response(
                        user_input, response)
                    st.write(html_response, unsafe_allow_html=True)

    def collection_tab(self) -> None:
        """
        Renders the Collection tab in the Streamlit app. Allows users to create a new collection by uploading PDF files.
        """
        new_collection_name = st.text_input(
            "Collection name", value="", key="new_collection_name")

        uploaded_files = st.file_uploader(
            "Upload a pdf or txt file",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.stop()

        if st.button("Create collection", type="secondary"):
            if new_collection_name:
                with st.spinner("Creating collection and feeding data. This may take a while..."):
                    # Save the files to disk
                    folder_path = save_files_to_disk(uploaded_files)
                    # Feed the files to ChromaDB
                    ChromaDB(self.embedding, self.chromadb_cli, new_collection_name).feed_from_path(
                        folder_path, data_type="pdf", split_documents=False)

                    st.success(
                        f"Collection ***{new_collection_name}*** created successfully.")
                    st.experimental_rerun()
            else:
                st.warning("Enter a collection name.")

    def run(self) -> None:
        """
        Runs the main application loop for Simple GPT.

        This method sets up the Streamlit page configuration, creates the sidebar, and displays two tabs for the user to interact with: "QA" and "New Collection". 
        The "QA" tab allows the user to ask questions and retrieve answers from the GPT model, while the "New Collection" tab allows the user to create a new collection 
        of questions and answers to be added to the model's training data.
        """
        # Config streamlit - Set the page title and icon
        st.set_page_config(page_title="Simple GPT",
                           page_icon="📖", layout="wide")

        self.sidebar()
        qa_t, collection_t = st.tabs(["QA", "New Collection"])

        with qa_t:
            qa_t.subheader("Retrieval QA")
            self.qa_tab()

        with collection_t:
            collection_t.subheader("Create a new collection")
            self.collection_tab()


if __name__ == "__main__":
    load_dotenv()

    # Using BGE model
    model_kwargs = {'device': get_device()}
    encode_kwargs = {'normalize_embeddings': True, "show_progress_bar": True}
    # set True to compute cosine similarity
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    # Using ChromaDB as vector database with a persistent storage
    client = chromadb.PersistentClient(path="db")

    # Initialize the app
    app = App(embedding_model, client)
    app.run()
