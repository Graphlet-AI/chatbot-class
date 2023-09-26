import logging
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
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
loader = PyPDFDirectoryLoader(PAPER_FOLDER)
docs = loader.load()

# Embed them with OpenAI ada model and store them in ChromaDB
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="data")
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
