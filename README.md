# AEB_RAG

## Enviroment

python: 3.11  
[LangChain](https://python.langchain.com/docs/introduction/)  
LLM: llama3.2(Ollama)  
Embedding Model: sentence-transformers/all-MiniLM-L6-v2(HuggingFaceEmbeddings)

## Result

[SElab_Industry_Academia_Collaboration](https://smith.langchain.com/public/4313b067-7883-4c73-b8f8-281d7ad9ba74/r)
[LangSmith](https://smith.langchain.com/public/dd752483-5f14-4f76-8336-29d35ae12802/r)

## 流程圖

### 開始 (Start)

- 初始化相關模型與工具：
  - 語言模型 (OllamaLLM)
  - 嵌入向量模型 (HuggingFaceEmbeddings)
  - 向量儲存 (InMemoryVectorStore)

### 加載資料 (Load Documents)

- 使用 PyPDFLoader 加載 PDF 檔案：docs/SElab_Industry_Academia_Collaboration.pdf。
- 返回的文檔被存儲於變數 docs。

### 資料處理

- 使用 RecursiveCharacterTextSplitter 將文檔分割為小片段：
  - 每片段大小：200 字符。
  - 重疊大小：40 字符。
- 分割後的片段存入變數 all_splits。
  索引建立 (Index Chunks)

- 將分割後的片段 (all_splits) 添加到向量儲存庫中 (vector_store.add_documents)。

- 狀態結構定義為：
  - question：用戶輸入的問題。
  - context：檢索到的相關文檔。
  - answer：生成的答案。

### 定義應用邏輯 (Application Logic)

- 檢索步驟 (Retrieve)
  - 根據 state["question"] 在向量儲存中進行相似性檢索。
  - 返回檢索結果作為上下文 context。
- 生成步驟 (Generate)
  - 將檢索到的文檔內容組合成一段文本。
  - 使用提示模板（prompt.invoke）生成消息。
  - 調用語言模型（llm.invoke）生成最終答案。

### 構建與編譯狀態圖 (Build and Compile StateGraph)

- 使用 StateGraph 定義應用步驟：
  - 添加步驟序列：retrieve -> generate。
  - 添加邊：START -> retrieve。
- 編譯狀態圖，生成可執行的應用邏輯。

## PDF
[SElab_Industry_Academia_Collaboration](docs/SElab_Industry_Academia_Collaboration.pdf)
