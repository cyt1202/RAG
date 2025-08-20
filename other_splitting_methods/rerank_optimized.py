import time
import os
from typing import List

# 新的分块器
from chonky import ParagraphSplitter

# LangChain 相关导入
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document

# --- 新增导入 ---
from openai import OpenAI
from sentence_transformers import CrossEncoder

'''
在chonky切割，faiss向量化，LLM增强查询（tokenlization_chonky.py)的基础上，
加入了cross-encoder进行重排，不过因为效果不明显，没有纳入最终的RAG的方法

'''
# --- 第一部分：使用 Chonky 进行文档处理  ---

def process_file_with_chonky(splitter: ParagraphSplitter, file_path: str, output_path: str) -> List[str]:
    """
    根据文件类型加载文件，使用 ParagraphSplitter 切分成文本块，并返回字符串列表。
    """
    print(f"\n--- 开始处理文件: {file_path} ---")
    start_time = time.time()
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到: {file_path}")
        return []

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
        full_text = "\n".join(doc.page_content for doc in documents)
        print(f" - 文件加载成功，总字数: {len(full_text)}")

    except Exception as e:
        print(f"错误：加载文件 {file_path} 失败: {e}")
        return []

    print("正在使用 Chonky ParagraphSplitter 进行文本切分...")
    chunks = list(splitter(full_text))
    print(f"文件处理完成，共切分出 {len(chunks)} 个文本块。")

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


# --- 第二部分：FAISS 向量数据库的创建与增强检索 ---

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

# --- LLM关键词提取 和 Cross-Encoder重排 ---

def extract_keywords_with_llm(client: OpenAI, query: str) -> str:
    """
    使用LLM（通义千问）提取查询中的核心关键词和概念。
    """
    print("\n--- 步骤 1: 使用 LLM 提取关键词 ---")
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个信息检索专家。请分析用户的查询，提取出最核心、最关键的专业术语和概念，用于向量数据库的搜索。不要回答问题，只需返回一个由逗号分隔的关键词列表。"},
                {"role": "user", "content": f"请为以下查询提取关键词：'{query}'"},
            ],
            temperature=0.0,
        )
        keywords = completion.choices[0].message.content.strip()
        print(f"原始查询: '{query}'")
        print(f"LLM 提取的关键词: '{keywords}'")
        # 将原始查询和关键词结合，构成更丰富的检索语句
        enhanced_query = f"{query} {keywords}"
        return enhanced_query
    except Exception as e:
        print(f"错误: 调用LLM API失败: {e}")
        # 如果API调用失败，则退回到原始查询
        return query

def rerank_with_cross_encoder(cross_encoder: CrossEncoder, query: str, docs: List[Document], k: int) -> List[Document]:
    """
    使用Cross-Encoder对召回的文档进行重排序。
    """
    print("\n--- 步骤 3: 使用 Cross-Encoder 进行重排 ---")
    start_time = time.time()
    
    # 1. 创建 [查询, 文档内容] 对
    pairs = [[query, doc.page_content] for doc in docs]
    
    # 2. 使用 cross-encoder 计算相关性分数
    scores = cross_encoder.predict(pairs)
    
    # 3. 将分数与文档绑定，并按分数降序排序
    doc_scores = list(zip(docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 4. 提取排序后的文档
    reranked_docs = [doc for doc, score in doc_scores]
    
    end_time = time.time()
    print(f"重排完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 5. 返回最终的 top-k 结果
    return reranked_docs[:k]


def enhanced_search_from_faiss(
    query: str, 
    embedding_model, 
    db_path: str, 
    # llm_client: OpenAI,
    cross_encoder: CrossEncoder,
    final_k: int = 3,
    retrieval_k: int = 10  # 初始召回数量，应大于 final_k
) -> List[Document]:
    """
    从FAISS向量数据库中加载，并执行 "LLM增强 -> 向量召回 -> Cross-Encoder精排" 的全流程检索。
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"向量数据库未找到: {db_path}。请先运行创建流程。")
    
    # 步骤 1: 使用LLM增强查询
    enhanced_query = extract_keywords_with_llm(llm_client, query)

    # 步骤 2: 从FAISS进行初步召回
    print(f"\n--- 步骤 2: 从 '{db_path}' 加载并进行向量召回 ---")
    # print(f"使用增强后查询进行相似性搜索: '{enhanced_query}'")
    start_time = time.time()
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    
    # 召回 retrieval_k 个候选文档
    retrieved_docs = db.similarity_search(enhanced_query, k=retrieval_k)
    retrieved_docs = db.similarity_search(query, k=retrieval_k)
    end_time = time.time()
    duration = end_time - start_time
    print(f"向量召回完成，找到 {len(retrieved_docs)} 个候选片段，耗时: {duration:.2f} 秒。")

    if not retrieved_docs:
        return []

    # 步骤 3: 使用Cross-Encoder进行重排
    final_results = rerank_with_cross_encoder(cross_encoder, query, retrieved_docs, final_k)

    print("\n--- 检索流程结束 ---")
    return final_results

# --- 主执行流程 ---

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    pdf_db_path = "vector_db/faiss_pdf_chonky_index"
    docx_db_path = "vector_db/faiss_word_chonky_index"
    pdf_file_path = "input/KG_test.pdf"
    docx_file_path = "input/sample.docx"
    pdf_output_path = "output/chonky_chunk_pdf.txt"
    docx_output_path = "output/chonky_chunk_word.txt"

    # 1. 确保目录存在
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("vector_db", exist_ok=True)

    # 2. 初始化模型
    print("正在初始化 Chonky ParagraphSplitter...")
    chonky_splitter = ParagraphSplitter(device="cpu") 
    print("ParagraphSplitter 初始化完成。")

    print("\n正在初始化用于FAISS的 LangChain Qwen 嵌入模型...")
    embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    qwen_embeddings_for_db = SentenceTransformerEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={},
        encode_kwargs={'prompt_name': "query"}
    )
    
    # --- 初始化 Cross-Encoder 和 LLM 客户端 ---
    print("\n正在初始化 Cross-Encoder 模型 (用于重排)...")
    # bge-reranker-base 是一个常用且效果很好的重排模型
    cross_encoder_model = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
    print("Cross-Encoder 初始化完成。")

    print("\n正在初始化 LLM 客户端 (用于查询增强)...")
    # 确保您已设置环境变量 DASHSCOPE_API_KEY
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError("错误: 请设置环境变量 DASHSCOPE_API_KEY")
    llm_client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print("LLM 客户端初始化完成。")

    # 3. 生成PDF向量库（逻辑与原来相同）
    if not os.path.exists(pdf_db_path):
        pdf_chunks_str = process_file_with_chonky(chonky_splitter, pdf_file_path, pdf_output_path)
        if pdf_chunks_str:
            print(f"\n正在将 {len(pdf_chunks_str)} 个文本块转换为Document对象...")
            pdf_docs = [Document(page_content=chunk) for chunk in pdf_chunks_str]
            create_faiss_vectorstore(pdf_docs, qwen_embeddings_for_db, pdf_db_path)
        else:
            print(f"\n警告: 未找到PDF文件 '{pdf_file_path}'，跳过PDF处理。")

    # 4. 生成Word向量库（逻辑与原来相同）
    if not os.path.exists(docx_db_path):
        docx_chunks_str = process_file_with_chonky(chonky_splitter, docx_file_path, docx_output_path)
        if docx_chunks_str:
            print(f"\n正在将 {len(docx_chunks_str)} 个文本块转换为Document对象...")
            docx_docs = [Document(page_content=chunk) for chunk in docx_chunks_str]
            create_faiss_vectorstore(docx_docs, qwen_embeddings_for_db, docx_db_path)
        else:
            print(f"\n警告: 未找到Word文件 '{docx_file_path}'，跳过Word处理。")

    # 5. 【增强版】检索PDF
    if os.path.exists(pdf_db_path):
        user_query_pdf = os.environ.get("USER_QUERY_PDF", "What is the Points Tracker?")
        print(f"\n{'='*20} 开始从PDF检索 {'='*20}")
        retrieved_docs = enhanced_search_from_faiss(
            query=user_query_pdf,
            embedding_model=qwen_embeddings_for_db,
            db_path=pdf_db_path,
            llm_client=llm_client,
            cross_encoder=cross_encoder_model,
            final_k=3, # 最终需要的结果数量
            retrieval_k=10 # 初始召回的候选数量
        )
        if retrieved_docs:
            print("\n--- 最终检索结果 ---")
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc.page_content}\n")
    
    # 6. 【增强版】检索Word
    if os.path.exists(docx_db_path):
        user_query_docx = os.environ.get("USER_QUERY_DOCX", "Tell me about Chronomirror")
        print(f"\n{'='*20} 开始从Word文档检索 {'='*20}")
        retrieved_docs = enhanced_search_from_faiss(
            query=user_query_docx,
            embedding_model=qwen_embeddings_for_db,
            db_path=docx_db_path,
            llm_client=llm_client,
            cross_encoder=cross_encoder_model,
            final_k=2
        )
        if retrieved_docs:
            print("\n--- 最终检索结果 ---")
            for i, doc in enumerate(retrieved_docs):
                print(f"--- 相关片段 {i+1} ---\n")
                print(f"内容: {doc.page_content}\n")

