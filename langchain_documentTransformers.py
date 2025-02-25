from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.schema import Document

# 使用 Hugging Face 嵌入模型
model_name = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

docs = [
    Document(page_content="這是一個測試文本。"),
    Document(page_content="這是一個測試文本！"),
    Document(page_content="這是一段完全不同的內容。"),
    Document(page_content="這是一個測試文本。")
]

# 使用嵌入去重
filter = EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=0.8)
filtered_docs = filter.transform_documents(docs)

# 輸出過濾後的文本
for d in filtered_docs:
    print(d.page_content)