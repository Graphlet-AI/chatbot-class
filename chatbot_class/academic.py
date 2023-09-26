import logging
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma

logging.getLogger("langchain").setLevel(logging.DEBUG)

# Dropbox folder with academic papers
PAPER_FOLDER = "/Users/rjurney/Dropbox/Academic Papers/"
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
embeddings = OpenAIEmbeddings(
    model="ada",
    max_retries=1,
    chunk_size=100,
)
fs = LocalFileStore("./data/embedding_cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, fs, namespace=embeddings.model)

vectordb = Chroma.from_documents(docs, embedding=cached_embedder, persist_directory="data")
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
