import time
import os
from typing import List

# 新的分块器
from semantic_text_splitter import TextSplitter

# LangChain 相关导入
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document  # <-- 关键导入，用于类型转换

'''
semantic-text-splitter
   3.1 在不超过指定长度（chunk_size）的前提下，尽可能让每个“区块”（chunk）包含最完整、最独立的语义单元

   3.2 分割器首先有一套预设的规则，定义了文本中不同元素的“重要性”或“层级”：标题 (Heading) > 段落 (Paragraph) > 句子 (Sentence) > 词 (Word) > 字符 (Character)

   3.3 对于MarkdownSplitter：这个层级非常清晰，可以直接利用了 Markdown 的语法结构；于 TextSplitter：它处理的是纯文本，所以使用换行符来模拟结构：最高级：连续的多个换行符（如 \n\n，通常代表段落分隔），次高级：单个换行符 (\n)，然后是：句子，最低级：单词和字符

   3.4 “ 自顶向下”寻找最合适的切分单位：一层一层寻找既能装进区块，又是最高级别的语义单位

   3.5 “贪心”合并，最大化区块长度：分割器就会开始“贪婪地”把相邻的段落一个个合并起来，直到快要超出 500 字符的限制为止

   3.6 它没有使用复杂的机器学习模型去做完美的句子切分，而是用了更简单、更高效的 Unicode 标准方法。这是出于性能考虑，这种“足够好”的方法在绝大多数情况下都能提供一个不错的语义断点，避免了运行缓慢的ML模型.
'''

# --- 第一部分：文档处理 (您的原代码) ---

def process_and_chunk_file(file_path: str, max_characters: int = 1024) -> List:
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
    print("正在进行文本切分...")
    splitter = TextSplitter(max_characters)
    full_text = "\n".join(doc.page_content for doc in documents)
    chunks = list(splitter.chunks(full_text))

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
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # 防止中文输出乱码（Python 3.7+）

    # 确保输入输出目录存在
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("vector_db", exist_ok=True)

    # --- 0. 初始化Qwen嵌入模型 ---
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model_kwargs = {}
    encode_kwargs = {'prompt_name': "query"}
    qwen_embeddings = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # --- 1. PDF向量库 ---
    pdf_file_path = "input/sample.pdf"
    pdf_db_path = "vector_db/faiss_pdf_splitter_index"
    if not os.path.exists(pdf_db_path):
        if os.path.exists(pdf_file_path):
            pdf_chunks = process_and_chunk_file(pdf_file_path)
            pdf_docs = [Document(page_content=chunk) for chunk in pdf_chunks]
            create_faiss_vectorstore(pdf_docs, qwen_embeddings, pdf_db_path)
        else:
            print(f"\n警告: 未找到PDF文件 '{pdf_file_path}'，跳过PDF处理。")

    # --- 2. Word向量库 ---
    docx_file_path = "input/sample.docx"
    docx_db_path = "vector_db/faiss_word_splitter_index"
    if not os.path.exists(docx_db_path):
        if os.path.exists(docx_file_path):
            docx_chunks = process_and_chunk_file(docx_file_path)
            docx_docs = [Document(page_content=chunk) for chunk in docx_chunks]
            create_faiss_vectorstore(docx_docs, qwen_embeddings, docx_db_path)
        else:
            print(f"\n警告: 未找到Word文件 '{docx_file_path}'，跳过Word处理。")

    # --- 3. 检索PDF ---
    if os.path.exists(pdf_db_path):
        user_query_pdf = os.environ.get("USER_QUERY_PDF", "Explain the method of MastSAM")
        retrieved_docs = search_from_faiss(user_query_pdf, qwen_embeddings, pdf_db_path, k=3)
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc.page_content}\n")

    # --- 4. 检索Word ---
    if os.path.exists(docx_db_path):
        user_query_docx = os.environ.get("USER_QUERY_DOCX", "What is Chronomirror")
        retrieved_docs = search_from_faiss(user_query_docx, qwen_embeddings, docx_db_path, k=3)
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc.page_content}\n")