import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate

from core.chroma import ChromaDB
from core.config import CONFIG, DEVICE
from core.files import save_files_to_disk
from core.text_utils import generate_html_response


@st.cache_resource
def instance_bge_model():
    """
    Loads a BGE model and returns a HuggingFaceBgeEmbeddings object.

    Returns:
        HuggingFaceBgeEmbeddings: An object that can be used to encode text into embeddings using the BGE model.
    """
    print("Loading BGE model...")
    # Using BGE model
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': True, "show_progress_bar": True}
    # set True to compute cosine similarity
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


@st.cache_resource
def instance_chroma_client():
    """
    Returns a ChromaDB client with a persistent storage.

    :return: ChromaDB PersistentClient object
    """
    print("Loading ChromaDB client...")
    # Using ChromaDB as vector database with a persistent storage
    return chromadb.PersistentClient(path="db")


@st.cache_resource
def instance_chain(
        _llm, _embedding, _chroma_cli, _prompt_template, llm_model_name, collection, prompt_name, k) -> BaseRetrievalQA:

    print(
        f"Instance OpenAI model: {_llm.model_name} with the collection: {collection} and prompt: {prompt_name}")

    chain_type_kwargs = {}
    if _prompt_template is not None:
        chain_type_kwargs = {"prompt": _prompt_template}

    return RetrievalQA.from_chain_type(
        llm=st.session_state['llm'],
        chain_type="stuff",
        retriever=ChromaDB(_embedding, _chroma_cli, collection).get_retriever(k=k),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

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
        self.chroma_cli = cli
        self.embedding = embedding
        self.k = CONFIG.get("k", 5)

        if 'collection_selected' not in st.session_state:
            st.session_state['collection_selected'] = None

        if 'prompt_template' not in st.session_state:
            st.session_state['prompt_template'] = None

        if 'retrieval_qa' not in st.session_state:
            st.session_state['retrieval_qa'] = None

        st.session_state["llm"] = ChatOpenAI(model_name=self.default_model_name)
        st.session_state['prompt_template_name'] = "default"

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

            st.session_state[key] = openai_key

            model_option = st.text_input(
                'Model name', value=self.default_model_name)

            if model_option:
                st.session_state['llm'] = ChatOpenAI(model_name=model_option)

            available_collections = []
            for collection in self.chroma_cli.list_collections():
                available_collections.append(collection.name)

            collection_option = st.selectbox(
                'Choose your collection', available_collections)

            if collection_option:
                st.session_state['collection_selected'] = collection_option
                if st.button("Delete collection", type="primary"):
                    self.chroma_cli.delete_collection(collection_option)
                    st.success(
                        f"Collection ***{collection_option}*** deleted successfully.")
                    st.experimental_rerun()

            available_prompts = ["default"]
            for prompt in CONFIG["prompts"]:
                available_prompts.append(prompt["name"])

            prompt_option = st.selectbox(
                'Choose your prompt', available_prompts)

            for prompt in CONFIG["prompts"]:
                if prompt["name"] == prompt_option:
                    st.session_state['prompt_template_name'] = prompt["name"]
                    st.session_state['prompt_template'] = PromptTemplate(
                        template=prompt["template"], input_variables=["context", "question"])

            with st.spinner("Loading parameters..."):
                # _llm, _embedding, _chroma_cli, _prompt_template, llm_model_name, collection, prompt_name, k
                chain = instance_chain(
                    st.session_state['llm'],
                    self.embedding,
                    self.chroma_cli,
                    st.session_state['prompt_template'],
                    model_option,
                    st.session_state['collection_selected'],
                    st.session_state['prompt_template_name'],
                    5,
                )

                st.session_state['retrieval_qa'] = chain

            st.markdown("---")
            st.markdown("# About")
            st.markdown(
                "SimpleGPT allows you to ask questions about your "
                "documents and get accurate answers while you can play with the prompt to get better results."
                "\n\n SimpleGPT: Use OpenAI's chat GPT models and BAAI/bge-base-en as embedding model."
            )


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

        if st.session_state['prompt_template']:
            with st.expander(f"See the selected prompt: {st.session_state['prompt_template_name']}"):
                st.markdown(
                    ":orange[If you want to change or create a prompt go to the config.yaml file.]")
                st.divider()
                st.write(st.session_state['prompt_template'].template)

        if not st.session_state['collection_selected']:
            st.warning(
                "Choose a collection in the sidebar or crate a new one."
            )

        if st.session_state['retrieval_qa'] is not None:
            user_input = st.text_input(
                f"Ask a question about your collection: ***{st.session_state['collection_selected']}***")

            if user_input:
                with st.spinner("Thinking..."):
                    response = st.session_state['retrieval_qa'](user_input)
                    html_response = generate_html_response(
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
                    ChromaDB(self.embedding, self.chroma_cli, new_collection_name).feed_from_path(
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
    # Config streamlit - Set the page title and icon
    st.set_page_config(page_title="Simple GPT", page_icon="📖", layout="wide")

    # Initialize the app
    app = App(instance_bge_model(), instance_chroma_client())
    app.run()
