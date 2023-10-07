import logging
import os
from typing import Any, Dict, List, Optional, Type

import chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.embeddings import Embeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma

logging.getLogger("langchain").setLevel(logging.DEBUG)

# Dropbox folder with academic papers
PAPER_FOLDER = "/home/rjurney/Dropbox/Academic Papers/"
assert os.path.exists(PAPER_FOLDER)

# Set in my ~/.zshrc
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Load all PDFs from academic paper folder
loader = PyPDFDirectoryLoader(PAPER_FOLDER, silent_errors=True)
docs = loader.load()

# How many papers on network motifs?
motif_docs = [(x.metadata["source"], x.page_content) for x in docs if "motif" in x.page_content]
motif_doc_count = len(motif_docs)
paper_count = len(set(x[0] for x in motif_docs))
print(
    f"You have {paper_count} papers on network motifs split across {motif_doc_count} document segments in `{PAPER_FOLDER}`."
)

# Embed them with OpenAI ada model and store them in ChromaDB
embeddings = OpenAIEmbeddings()# openai_api_key=os.environ["OPENAI_API_KEY"])
fs = LocalFileStore("./data/embedding_cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, fs, namespace=embeddings.model)


class RobustChroma(Chroma):
    """Handle UnicodeDecodeErrors and don't die, just skip them."""

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "langchain",
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # Add this line
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """

        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )

        last_file = None
        texts, metadatas = [], []
        for doc in documents:
            filename = doc.metadata["source"]
            try:
                if filename != last_file:
                    if last_file:
                        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                    texts, metadatas = [], []
                    last_file = filename

                # Build up a cache of documents to add...
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            except UnicodeDecodeError:
                logging.warning(
                    f'Skipping document due to UnicodeDecodeError: {doc.metadata["source"]}'
                )
                continue  # Skip to the next document

        return chroma_collection

    @staticmethod
    def partition(lst, batch_size):
        """Partition a list into batches of a specified size.

        Args:
            lst (list): The list to partition.
            batch_size (int): The size of each batch.

        Returns:
            list: A list of lists, where each inner list is a batch.
        """
        # For input list of length 0-1, return the list itself within another list
        if len(lst) <= 1:
            return [lst]
        # For other cases, create the batches
        return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


vectordb = RobustChroma.from_documents(docs, embedding=cached_embedder, persist_directory="data")
vectordb.persist()

# Setup a simple buffer memory system to submit with the API calls to provide prompt context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a ConversationalRetrievalChain from the LLM, the vectorstore, and the memory system
qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0.8),
    vectordb.as_retriever(),
    memory=memory,
    verbose=True,
)

result = qa({"question": "What are the different types of network motif?"})
print(result)
