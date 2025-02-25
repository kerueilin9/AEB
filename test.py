import os

LANGCHAIN_TRACING_V2='true'
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"# Place your API Key
LANGCHAIN_PROJECT="RAG_SELAB"

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY 
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

from langchain_ollama import OllamaLLM
# model = "llama3.2"
model = "mistral"
llm = OllamaLLM(model=model, base_url="http://localhost:11434")

from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

import bs4
from langchain_community.document_loaders import WebBaseLoader

# Load and chunk contents of the blog
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/SElab_Industry_Academia_Collaboration.pdf")
docs = loader.load()

from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "產學是什麼"})
print(response["answer"])