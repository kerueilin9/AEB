import os
import uuid
import base64
import requests
from IPython import display
from unstructured.partition.pdf import partition_pdf
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever

import os

LANGCHAIN_TRACING_V2='true'
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"# Place your API Key
LANGCHAIN_PROJECT="RAG_SELAB"

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY 
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# os.environ['OPENAI_API_KEY'] = "sk-*****"

from langchain_ollama import OllamaLLM
# model = "llama3.2"
model = "mistral"
llm = OllamaLLM(model=model, base_url="http://localhost:11434")

output_path = "./images/Swin"

# Get elements
raw_pdf_elements = partition_pdf(
    filename="./docs/swin.pdf",
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    extract_image_block_output_dir=output_path,
)

# Get text summaries and table summaries
text_elements = []
table_elements = []

text_summaries = []
table_summaries = []

summary_prompt = """
Summarize the following {element_type}: 
{element}
"""
summary_chain = PromptTemplate.from_template(summary_prompt) | llm

for e in raw_pdf_elements:
    if 'CompositeElement' in repr(e):
        text_elements.append(e.text)
        summary = summary_chain.invoke({'element_type': 'text', 'element': e})
        text_summaries.append(summary)

    elif 'Table' in repr(e):
        table_elements.append(e.text)
        summary = summary_chain.invoke({'element_type': 'table', 'element': e})
        table_summaries.append(summary)
        
# Get image summaries
image_elements = []
image_summaries = []

# LLaVA
def summarize_image(encoded_image):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava",  
        "prompt": "Describe the contents of this image.",
        "stream": False,
        "images": [encoded_image]
    }
    response = requests.post(url, json=payload)
    return response.json()["response"]

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
    
for i in os.listdir(output_path):
    if i.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(output_path, i)
        encoded_image = encode_image(image_path)
        image_elements.append(encoded_image)
        summary = summarize_image(encoded_image)
        image_summaries.append(summary)
        
# Create Documents and Vectorstore
documents = []
retrieve_contents = []

for e, s in zip(text_elements, text_summaries):
    i = str(uuid.uuid4())
    doc = Document(
        page_content = s,
        metadata = {
            'id': i,
            'type': 'text',
            'original_content': e
        }
    )
    retrieve_contents.append((i, e))
    documents.append(doc)
    
for e, s in zip(table_elements, table_summaries):
    doc = Document(
        page_content = s,
        metadata = {
            'id': i,
            'type': 'table',
            'original_content': e
        }
    )
    retrieve_contents.append((i, e))
    documents.append(doc)
    
for e, s in zip(image_elements, image_summaries):
    doc = Document(
        page_content = s,
        metadata = {
            'id': i,
            'type': 'image',
            'original_content': e
        }
    )
    retrieve_contents.append((i, s))
    documents.append(doc)

from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-m3"

vectorstore = FAISS.from_documents(documents=documents, embedding=HuggingFaceEmbeddings(model_name=model_name))

answer_template = """
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question} 
"""
answer_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(answer_template))

def answer(question):
    relevant_docs = vectorstore.similarity_search(question)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = answer_chain.run({'context': context, 'question': question})
    return result, relevant_images

# Display result
result, relevant_images = answer("What is the difference between Swin Transformer and ViT?")
print(result)
for e in relevant_images:
    display.display(display.Image(b64decode(e)))