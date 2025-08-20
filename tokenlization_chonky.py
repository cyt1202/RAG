import time
import os
from typing import List
import argparse

# 新的分块器
from chonky import ParagraphSplitter

# LangChain 相关导入
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document  # <-- 关键导入，用于类型转换

# --- 第一部分：使用 Chonky 进行文档处理 ---

def process_file_with_chonky(splitter: ParagraphSplitter, file_path: str, output_path: str) -> List[str]:
    """
    根据文件类型加载文件，使用 ParagraphSplitter 切分成文本块，并返回字符串列表。

    参数:
        splitter (ParagraphSplitter): Chonky 分割器实例.
        file_path (str): Word 或 PDF 文件的路径.
        output_path (str): 分块结果的保存路径.

    返回:
        List[str]: 切分后的文本块字符串列表.
    """
    print(f"\n--- 开始处理文件: {file_path} ---")
    start_time = time.time()
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到: {file_path}")
        return []

    # 1. 根据文件扩展名选择合适的加载器
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}。仅支持 .pdf 和 .docx。")

        print(f"正在加载文件: {file_path}...")
        documents = loader.load()
        # 将所有页面的内容合并成一个长文本
        full_text = "\n".join(doc.page_content for doc in documents)
        print(f" - 文件加载成功，总字数: {len(full_text)}")

    except Exception as e:
        print(f"错误：加载文件 {file_path} 失败: {e}")
        return []

    # 2. 使用 Chonky ParagraphSplitter 进行文本切分
    print("正在使用 Chonky ParagraphSplitter 进行文本切分...")
    chunks = list(splitter(full_text))
    print(f"文件处理完成，共切分出 {len(chunks)} 个文本块。")

    # 3. 将分块结果写入输出文件以供检查
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk + "\n\n")
        print(f" - 切分结果已成功写入到: {output_path}")
    except Exception as e:
        print(f"错误：写入文件 {output_path} 失败: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"--- 处理 {file_path} 总耗时: {duration:.2f} 秒 ---")
    return chunks

# --- 第二部分：FAISS 向量数据库的创建与检索  ---

def create_faiss_vectorstore(docs: List[Document], embedding_model, db_path: str):
    """
    使用指定的嵌入模型将Document列表向量化，并创建/保存一个FAISS向量数据库。
    """
    print(f"\n正在使用Qwen模型创建向量数据库...")
    start_time = time.time()
    db = FAISS.from_documents(docs, embedding_model)
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

    #query 是不是应该先用llm提取关键词再进行相似度检索！！！
    #直接similarity是不是有点粗糙？使用cross-encoder会不会好点？
    results = db.similarity_search(query, k=k)
    end_time = time.time()
    duration = end_time - start_time
    print(f"--- 检索耗时: {duration:.2f} 秒 ---")
    return results
    #CHROMEDB

# --- 主执行流程 ---
#运行指令 python tokenlization_chonky.py --pdf_file input/your.pdf --docx_file input/your.docx --pdf_query "你的PDF问题" --docx_query "你的Word问题"
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # 使用 argparse 来处理命令行参数；也可以直接在代码中设置默认值
    #===================================================================
    # ================在代码中直接设置输入文件路径和查询内容================
    #===================================================================
    parser = argparse.ArgumentParser(description="Chonky 文档分块与检索")
    parser.add_argument("--pdf_file", type=str, default="input/KG_test.pdf", help="PDF文件路径")
    parser.add_argument("--docx_file", type=str, default="input/sample.docx", help="Word文件路径")

    parser.add_argument("--docx_output", type=str, default="output/chonky_chunk_word.txt", help="Word分块输出路径")
    parser.add_argument("--pdf_output", type=str, default="output/chonky_chunk_pdf.txt", help="PDF分块输出路径")

    parser.add_argument("--pdf_db_path", type=str, default="vector_db/faiss_pdf_chonky_index", help="PDF向量数据库路径")
    parser.add_argument("--docx_db_path", type=str, default="vector_db/faiss_word_chonky_index", help="Word向量数据库路径")

    parser.add_argument("--pdf_query", type=str, default="What is the method of MastSAM?", help="PDF检索query")
    parser.add_argument("--docx_query", type=str, default="What is Chronomirror", help="Word检索query")
    args = parser.parse_args()

    pdf_file_path = args.pdf_file
    docx_file_path = args.docx_file
    user_query_pdf = args.pdf_query
    user_query_docx = args.docx_query
    pdf_output_path = args.pdf_output
    docx_output_path = args.docx_output

    #=====================================================================================================
    #=======改了输入文件记得把输出的向量数据库和分词文件都删掉，这个算法会看如果这两个文件存在就不会重新生成=======
    #=====================================================================================================
    pdf_db_path = args.pdf_db_path
    docx_db_path = args.docx_db_path


    # 1. 确保目录存在
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("vector_db", exist_ok=True)

    # 2. 初始化模型
    print("正在初始化 Chonky ParagraphSplitter...")
    # 首次运行会下载模型
    chonky_splitter = ParagraphSplitter(device="cpu") 
    print("ParagraphSplitter 初始化完成。")

    print("\n正在初始化用于FAISS的 LangChain Qwen 嵌入模型...")
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    qwen_embeddings_for_db = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs={},
        encode_kwargs={'prompt_name': "query"}
    )

    # 3. 生成PDF向量库 ；
    #===========================================================================================================
    #============注意！如果向量库已经存在，这一步会被跳过；所以如果输入文档变化，需要手动删除旧的向量库文件=============
    #===========================================================================================================
    if not os.path.exists(pdf_db_path):
        pdf_chunks_str = process_file_with_chonky(chonky_splitter, pdf_file_path, pdf_output_path)
        
        if pdf_chunks_str:
            # 步骤 2: 【关键步骤】将字符串块转换为 LangChain Document 对象
            print(f"\n正在将 {len(pdf_chunks_str)} 个文本块转换为Document对象...")
            pdf_docs = [Document(page_content=chunk) for chunk in pdf_chunks_str]
            
            # 步骤 3: 创建并保存FAISS数据库
            create_faiss_vectorstore(pdf_docs, qwen_embeddings_for_db, pdf_db_path)
        else:
            print(f"\n警告: 未找到PDF文件 '{pdf_file_path}'，跳过PDF处理。")

    # 4. 生成Word向量库（如不存在）
    if not os.path.exists(docx_db_path):
        # 步骤 1: 使用 Chonky 分割器获取字符串块
        docx_chunks_str = process_file_with_chonky(chonky_splitter, docx_file_path, docx_output_path)
        
        if docx_chunks_str:
            # 步骤 2: 【关键步骤】将字符串块转换为 LangChain Document 对象
            print(f"\n正在将 {len(docx_chunks_str)} 个文本块转换为Document对象...")
            docx_docs = [Document(page_content=chunk) for chunk in docx_chunks_str]

            # 步骤 3: 创建并保存FAISS数据库
            create_faiss_vectorstore(docx_docs, qwen_embeddings_for_db, docx_db_path)
        else:
            print(f"\n警告: 未找到Word文件 '{docx_file_path}'，跳过Word处理。")

    # 5. 检索PDF
    if os.path.exists(pdf_db_path):
        user_query_pdf = args.pdf_query
        retrieved_docs = search_from_faiss(user_query_pdf, qwen_embeddings_for_db, pdf_db_path, k=5)
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc.page_content}\n")

    # 6. 检索Word
    if os.path.exists(docx_db_path):
        user_query_docx = args.docx_query
        retrieved_docs = search_from_faiss(user_query_docx, qwen_embeddings_for_db, docx_db_path, k=3)
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc.page_content}\n")


