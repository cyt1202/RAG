import time
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_splitter import SentenceSplitter
from typing import List

# LangChain 相关导入
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document  # <--- 关键导入

from sklearn.metrics.pairwise import cosine_similarity

'''
自己尝试的一个切分方法：
使用 Qwen3-Embedding-0.6B 模型进行语义分割: 
1. 计算相邻句子的余弦相似度，如果句子间的相似度差距大于阈值(0.5),则认为是一个语义断点。
2. 将当前句子加入缓冲区
3. 将当前缓冲区中的句子拼接成临时文本，用于计算长度，当到达最后一个句子或语义断点时，检查当前块的长度，
    如果当前块的长度超过最小块大小，则将其保存为一个块
4.输出

结果：表现一般,即使有不步骤三的合并句子的逻辑，仍然会有很多小块。切的太碎了。

'''

class QwenSemanticSplitter:
    """
    使用 Qwen3-Embedding-0.6B 模型和 sentence-transformers 实现语义分割。
    """
    def __init__(self, model_name:str = "Qwen/Qwen3-Embedding-0.6B" , 
                 breakpoint_percentile_threshold: int = 95,
                 min_chunk_size: int = 256):
        print("QwenSemanticSplitter 使用已加载的模型实例进行初始化...")
        # 直接使用传入的模型实例，不再自己加载
        self.model = SentenceTransformer(model_name)
        self.sentence_splitter = SentenceSplitter(language='en')
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.min_chunk_size = min_chunk_size
        print("分割器初始化完成。")

    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """为句子列表生成嵌入向量。"""
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        return embeddings

    def split(self, text: str) -> List[str]:
        """对输入的长文本进行语义切分。"""
        sentences = self.sentence_splitter.split(text)
        if len(sentences) <= 1:
            return sentences

        print(f" - 文本被分割成 {len(sentences)} 个句子。")
        embeddings = self._get_embeddings(sentences)
        print(" - 句子嵌入向量生成完毕。")

        # 计算相邻句子的余弦相似度
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()
        print(" - 相邻句子相似度计算完毕。")

        # 使用numpy的percentile函数确定断点阈值
        breakpoint_threshold = np.percentile(similarities, self.breakpoint_percentile_threshold)
        print(f" - 根据 {self.breakpoint_percentile_threshold} 百分位，确定相似度断点阈值为: {breakpoint_threshold:.3f}")
        
        # 找出相似度低于阈值的位置作为断点
        breakpoint_indices = [i for i, sim in enumerate(similarities) if sim < breakpoint_threshold]
        print(f" - 找到 {len(breakpoint_indices)} 个语义断点。")

        chunks = []
        current_chunk_sentences = [] # 当前正在构建的块所包含的句子
        
        for i, sentence in enumerate(sentences):
            # 将当前句子加入缓冲区
            current_chunk_sentences.append(sentence)
            
            # 判断是否到达最后一个句子
            is_last_sentence = (i == len(sentences) - 1)
            
            # 判断当前句子和下一个句子之间是否存在语义断点
            # 如果是最后一个句子，则认为它后面有一个“强制”断点
            is_breakpoint = (i < len(similarities) and similarities[i] < breakpoint_threshold)
            
            # 将当前缓冲区中的句子拼接成临时文本，用于计算长度
            current_chunk_text = " ".join(current_chunk_sentences)

            # 决策点：何时将缓冲区的内容输出为一个块？
            # 条件1：到达了最后一个句子（必须输出）
            # 条件2：遇到了一个语义断点，并且当前块的长度已经超过了设定的最小长度
            if is_last_sentence or (is_breakpoint and len(current_chunk_text) >= self.min_chunk_size):
                chunks.append(current_chunk_text)
                # 清空缓冲区，为下一个块做准备
                current_chunk_sentences = []

        print(f" - 智能合并切分完成，共生成 {len(chunks)} 个块。")
        return chunks

def process_file(splitter: QwenSemanticSplitter, file_path: str, output_path: str) -> List[str]:
    """加载、处理单个文件并返回文本块列表"""
    print(f"\n--- 开始处理文件: {file_path} ---")
    start_time = time.time()

    full_text = ""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")

        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])

    except Exception as e:
        print(f"错误：加载文件 {file_path} 失败: {e}")
        return []

    print(f" - 文件加载成功，总字数: {len(full_text)}")

    # 执行语义切分
    chunks = splitter.split(full_text)
    
    # 写入输出文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"源文件: {file_path}\n")
            f.write(f"语义切分结果 (共 {len(chunks)} 个块):\n")
            f.write("="*40 + "\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk.strip())
                f.write("\n\n" + "="*40 + "\n\n")
        print(f" - 切分结果已成功写入到: {output_path}")
    except Exception as e:
        print(f"错误：写入文件 {output_path} 失败: {e}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"--- 处理 {file_path} 总耗时: {duration:.2f} 秒 ---")
    return chunks

def create_faiss_vectorstore(docs: List[Document], embedding_model, db_path: str):
    """
    使用指定的嵌入模型将Document列表向量化，并创建/保存一个FAISS向量数据库。
    """
    print(f"\n正在使用Qwen模型创建向量数据库...")
    start_time = time.time()
    
    # 从文档列表和嵌入模型创建FAISS向量存储
    db = FAISS.from_documents(docs, embedding_model)
    
    # 将数据库保存到本地
    db.save_local(db_path)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"向量数据库已成功创建并保存到 '{db_path}'。")
    print(f"--- 创建和保存耗时: {duration:.2f} 秒 ---")

def search_from_faiss(query: str, embedding_model, db_path: str, k: int = 3) -> List[Document]:
    """
    从FAISS向量数据库中加载并检索与查询最相关的文档。
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"向量数据库未找到: {db_path}。请先运行创建流程。")
        
    print(f"\n正在从 '{db_path}' 加载向量数据库...")
    start_time = time.time()
    
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    
    print(f"正在执行相似性搜索，查询: '{query}'")
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
    
    # --- 0. 初始化模型 ---
    print("正在初始化 Qwen 语义分割器...")

     # 您可以调整 breakpoint_percentile_threshold 来控制切分粒度

     # 较高的值（如95）会产生较大的块，较低的值（如85）会产生较小的块
    qwen_splitter = QwenSemanticSplitter(breakpoint_percentile_threshold=90)
    print("\n正在初始化用于FAISS的 LangChain Qwen 嵌入模型...")
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    # LangChain的包装器，用于FAISS数据库
    qwen_embeddings_for_db = SentenceTransformerEmbeddings(
    model_name=model_name,
    model_kwargs={},
    encode_kwargs={'prompt_name': "query"}

    )

    print("所有模型初始化完成。")

    # --- 3. 处理PDF文件 ---
    pdf_file_path = "input/sample.pdf"
    pdf_output_path = "output/Qwen_chunk_pdf.txt"
    pdf_db_path = "vector_db/faiss_pdf_index"
    
    if os.path.exists(pdf_file_path):
        # 步骤 1: 使用语义分割器获取字符串块
        pdf_chunks_str = process_file(qwen_splitter, pdf_file_path, pdf_output_path)
        
        if pdf_chunks_str:
            # 步骤 2: 【关键】将字符串块转换为 LangChain Document 对象
            print(f"\n正在将 {len(pdf_chunks_str)} 个文本块转换为Document对象...")
            pdf_docs = [Document(page_content=chunk) for chunk in pdf_chunks_str]
            
            # 步骤 3: 创建并保存FAISS数据库
            create_faiss_vectorstore(pdf_docs, qwen_embeddings_for_db, pdf_db_path)
    else:
        print(f"\n警告: 未找到PDF文件 '{pdf_file_path}'，跳过PDF处理。")

    # --- 4. 处理Word文件 ---
    docx_file_path = "input/sample.docx"
    docx_output_path = "output/Qwen_chunk_word.txt" # <--- 修正了输出路径
    docx_db_path = "vector_db/faiss_word_index"

    if os.path.exists(docx_file_path):
        # 步骤 1: 使用语义分割器获取字符串块
        docx_chunks_str = process_file(qwen_splitter, docx_file_path, docx_output_path) # <--- 修正了文件和输出路径
        
        if docx_chunks_str:
            # 步骤 2: 【关键】将字符串块转换为 LangChain Document 对象
            print(f"\n正在将 {len(docx_chunks_str)} 个文本块转换为Document对象...")
            docx_docs = [Document(page_content=chunk) for chunk in docx_chunks_str]

            # 步骤 3: 创建并保存FAISS数据库
            create_faiss_vectorstore(docx_docs, qwen_embeddings_for_db, docx_db_path)
    else:
        print(f"\n警告: 未找到Word文件 '{docx_file_path}'，跳过Word处理。")


    # --- 5. 执行检索 ---
    print("\n" + "="*50)
    print("所有文件处理完毕，现在开始执行检索示例。")
    print("="*50)

    # 示例 1: 在PDF的向量数据库中进行检索
    if os.path.exists(pdf_db_path):
        user_query_pdf = "What is ICE?" # 请根据你的PDF文档内容修改问题
        retrieved_docs = search_from_faiss(user_query_pdf, qwen_embeddings_for_db, pdf_db_path, k=3)
        
        print("\n--- PDF检索结果 ---\n")
        if not retrieved_docs:
            print("没有找到相关内容。")
        else:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc.page_content}\n")

    # 示例 2: 在Word的向量数据库中进行检索
    if os.path.exists(docx_db_path):
        user_query_docx = "What is Chronomirror" # 请根据你的Word文档内容修改问题
        retrieved_docs = search_from_faiss(user_query_docx, qwen_embeddings_for_db, docx_db_path, k=3)
        
        print("\n--- Word检索结果 ---\n")
        if not retrieved_docs:
            print("没有找到相关内容。")
        else:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc.page_content}\n")