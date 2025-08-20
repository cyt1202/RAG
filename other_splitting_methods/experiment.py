import time
import os
from typing import List

# 新的分块器
from semantic_text_splitter import TextSplitter
from chonky import ParagraphSplitter

# LangChain 相关导入
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document  # <-- 关键导入，用于类型转换

# --- 第一部分：文档处理 (您的原代码) ---

def process_and_chunk_file(file_path: str, max_characters: int = 1024, method: str = chronky) -> List:
    """
    根据文件类型加载文件，并将其切分成文本块。

    参数:
        file_path (str): Word 或 PDF 文件的路径.
        chunk_size (int): 每个文本块的最大字符数.
        chunk_overlap (int): 文本块之间的重叠字符数.

    返回:
        List[Document]: 切分后的 LangChain Document 对象列表.
    """
    start_time = time.time()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    # 根据文件扩展名选择合适的加载器
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_extension}。仅支持 .pdf 和 .docx。")

    print(f"正在加载文件: {file_path}...")
    documents = loader.load()


    # ======================================更换分块器（input output均为str）======================================
    if method == splitter:
        print("正在进行文本切分...")
        text_splitter = TextSplitter(max_characters)
        full_text = "\n".join(doc.page_content for doc in documents)
        chunks = list(text_splitter.chunks(full_text))

    elif method == chonky:
        full_text = "\n".join(doc.page_content for doc in documents)
        chunks = list(splitter(full_text))

    else:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, #有点难改
        chunk_overlap = chunk_overlap, #有点难改
    )
    chunks = text_splitter.split_documents(documents)
    # ======================================更换分块器（input output均为str）======================================
    
    if file_extension == ".pdf":
        with open("output/Splitter_chunk_pdf.txt", "w", encoding="utf-8") as f:
            for i,chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk + "\n\n")

    elif file_extension == ".docx":
        with open("output/Splitter_chunk_word.txt", "w", encoding="utf-8") as f:
            for i,chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk + "\n\n")
    
    print(f"文件处理完成，共切分出 {len(chunks)} 个文本块。")
    end_time = time.time()
    duration = end_time - start_time
    print(f"--- 处理 {file_path} 总耗时: {duration:.2f} 秒 ---")
    return chunks

# --- 第二部分：向量化与存储 ---

def create_faiss_vectorstore(docs: List[Document], embedding_model, db_path: str):
    """
    使用指定的嵌入模型将文本块向量化，并创建/保存一个FAISS向量数据库。

    参数:
        chunks (List): LangChain Document 对象列表.
        embedding_model: 用于向量化的嵌入模型实例.
        db_path (str): FAISS 索引的保存路径.
    """
    print(f"\n正在使用Qwen模型创建向量数据库...")
    start_time = time.time()
    
    # 从文档块和嵌入模型创建FAISS向量存储
    # LangChain的FAISS.from_documents会自动处理向量化过程
    db = FAISS.from_documents(docs, embedding_model)
    
    # 将数据库保存到本地
    db.save_local(db_path)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"向量数据库已成功创建并保存到 '{db_path}'。")
    print(f"--- 创建和保存耗时: {duration:.2f} 秒 ---")

# --- 第三部分：检索 ---

def search_from_faiss(query: str, embedding_model, db_path: str, k: int = 3) -> List:
    """
    从FAISS向量数据库中加载并检索与查询最相关的文档。

    参数:
        query (str): 用户的查询字符串.
        embedding_model: 用于向量化的嵌入模型实例 (必须与创建时相同).
        db_path (str): FAISS 索引的加载路径.
        k (int): 要检索的最相关文档数量.

    返回:
        List[Document]: 检索到的 LangChain Document 对象列表.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"向量数据库未找到: {db_path}。请先运行创建流程。")
        
    print(f"\n正在从 '{db_path}' 加载向量数据库...")
    start_time = time.time()
    
    # 加载本地数据库，注意需要提供相同的嵌入模型
    # allow_dangerous_deserialization=True 是加载本地FAISS索引所必需的
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    
    print(f"正在执行相似性搜索，查询: '{query}'")
    # 使用similarity_search方法进行检索
    results = db.similarity_search(query, k=k)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"--- 检索耗时: {duration:.2f} 秒 ---")
    
    return results


# --- 主执行流程 ---

if __name__ == "__main__":
    # 确保输入输出目录存在
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("vector_db", exist_ok=True)
    
    # --- 0. 初始化Qwen嵌入模型 ---
    # 我们使用LangChain的包装器，它能在后台高效地处理SentenceTransformer模型
    print("正在初始化 Qwen/Qwen3-Embedding-0.6B 模型...")
    # 定义模型名称
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    # 定义查询指令，LangChain的包装器会自动处理
    # 注意：这里的 'instruction' 参数名对应 SentenceTransformerEmbeddings 的实现
    model_kwargs = {}
    encode_kwargs = {
        'prompt_name': "query", # 使用SentenceTransformer模型中为查询定义的prompt
    }

    qwen_embeddings = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # 对于Qwen模型，它内部已经定义好了prompt，我们通过 `prompt_name` 来调用
    # `SentenceTransformerEmbeddings` 会在调用 `embed_query` 时自动应用这个prompt
    
    print("模型初始化完成。")


    # --- 1. 处理PDF文件 ---
    pdf_file_path = "input/sample.pdf" # 请确保这个文件存在

    pdf_db_path = "vector_db/faiss_pdf_index"
    
    # 检查PDF文件是否存在，如果不存在则跳过
    if os.path.exists(pdf_file_path):
        pdf_chunks = process_and_chunk_file(pdf_file_path)
        pdf_docs = [Document(page_content=chunk) for chunk in pdf_chunks]
        create_faiss_vectorstore(pdf_docs, qwen_embeddings, pdf_db_path)
    else:
        print(f"\n警告: 未找到PDF文件 '{pdf_file_path}'，跳过PDF处理。")

    # --- 2. 处理Word文件 ---
    docx_file_path = "input/sample.docx" # 请确保这个文件存在
    docx_db_path = "vector_db/faiss_word_index"

    # 检查Word文件是否存在，如果不存在则跳过
    if os.path.exists(docx_file_path):
        docx_chunks = process_and_chunk_file(docx_file_path)
        docx_docs = [Document(page_content=chunk) for chunk in docx_chunks]
        create_faiss_vectorstore(docx_docs, qwen_embeddings, docx_db_path)
    else:
        print(f"\n警告: 未找到Word文件 '{docx_file_path}'，跳过Word处理。")


    # --- 3. 执行检索 ---
    # 假设我们要在PDF的向量数据库中进行检索
    if os.path.exists(pdf_db_path):
        # 你的问题
        user_query = "What is ICE?" # 请根据你的文档内容修改问题
        
        retrieved_docs = search_from_faiss(user_query, qwen_embeddings, pdf_db_path, k=3)
        
        print("\n--- 检索结果 ---\n")
        if not retrieved_docs:
            print("没有找到相关内容。")
        else:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc}\n")
                # 如果需要，可以打印元数据，如来源页码
                if 'page' in doc.metadata:
                    print(f"来源页码: {doc.metadata['page']}\n")

    if os.path.exists(docx_db_path):
        # 你的问题
        user_query = "What is Chronomirror" # 请根据你的文档内容修改问题
        retrieved_docs = search_from_faiss(user_query, qwen_embeddings, docx_db_path, k=3)
        print("\n--- 检索结果 ---\n")
        if not retrieved_docs:
            print("没有找到相关内容。")
        else:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc}\n")
                # 如果需要，可以打印元数据，如来源页码
                if 'page' in doc.metadata:
                    print(f"来源页码: {doc.metadata['page']}\n")
    else:
        print("\n无法执行检索，因为向量数据库不存在。")