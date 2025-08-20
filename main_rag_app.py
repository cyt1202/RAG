"""
混合检索RAG系统的统一命令行工具 (Hybrid Retrieval RAG System CLI)

本脚本整合了知识图谱构建、向量数据库创建和混合检索问答的全流程。

功能:
1.  根据输入文档，自动构建并缓存知识图谱 (JSON格式)。
2.  根据输入文档，自动进行语义切分、向量化，并构建/缓存FAISS向量数据库。
3.  执行混合检索，结合知识图谱(精确事实)和向量检索(语义上下文)来回答用户问题。
4.  支持通过命令行参数指定输入/输出文件和问题，也支持零参数使用默认值运行。
5.  支持强制重建知识图谱和向量库。
"""
import os
import json
import sys
import argparse
import time
import shutil
from typing import List, Dict

# LangChain & AI 模型相关导入
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 新的分块器
from chonky import ParagraphSplitter

# --- 0. 环境和全局模型初始化 ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    if not DASHSCOPE_API_KEY:
        print("警告：未找到'DASHSCOPE_API_KEY'环境变量。部分功能可能受限。")
        sys.exit(1)
except ImportError:
    print("警告: python-dotenv未安装，无法从.env文件加载API Key。")
    sys.exit(1)

# 初始化用于答案生成和图谱构建的LLM
# 注意：这里我们统一使用ChatOpenAI接口，因为它在之前的测试中更稳定
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    max_tokens=4096, # 保证有足够的空间生成图谱和答案
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
print("✅ LLM (qwen-plus) 初始化成功。")

# 初始化用于向量化的嵌入模型
embedding_model = SentenceTransformerEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={},
    encode_kwargs={'prompt_name': "query"}
)
print("✅ 嵌入模型 (Qwen3-Embedding-0.6B) 初始化成功。")


# --- 1. 数据处理与构建函数 ---

def build_knowledge_graph(llm_for_kg: ChatOpenAI, input_file: str, output_kg_path: str):
    """从文档构建知识图谱并保存为JSON。"""
    print(f"\n--- 正在为 '{os.path.basename(input_file)}' 构建知识图谱 ---")
    start_time = time.time()
    
    # 加载文档
    file_extension = os.path.splitext(input_file)[1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(input_file)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(input_file)
    else:
        raise ValueError(f"不支持的文件类型: {file_extension}。")
    
    docs = loader.load()
    full_text = "\n".join(doc.page_content for doc in docs)
    documents = [Document(page_content=full_text)]

    # 使用LLMGraphTransformer构建图谱
    # 注意：ChatTongyi在之前测试中存在问题，这里我们统一使用更稳定的ChatOpenAI接口
    transformer = LLMGraphTransformer(llm=llm_for_kg)
    graph_documents = transformer.convert_to_graph_documents(documents)

    # 保存图谱数据
    print(f"正在将知识图谱写入: {output_kg_path}")
    output_data = [{
        "nodes": [node.dict() for node in doc.nodes],
        "relationships": [rel.dict() for rel in doc.relationships],
        "source": doc.source.dict()
    } for doc in graph_documents]
    
    with open(output_kg_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    duration = time.time() - start_time
    print(f"✅ 知识图谱构建完成，耗时: {duration:.2f} 秒。")

def build_vector_store(embedding_model, input_file: str, output_db_path: str):
    """对文档进行切分、向量化并构建FAISS数据库。"""
    print(f"\n--- 正在为 '{os.path.basename(input_file)}' 构建向量数据库 ---")
    start_time = time.time()

    # 加载文档
    file_extension = os.path.splitext(input_file)[1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(input_file)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(input_file)
    else:
        raise ValueError(f"不支持的文件类型: {file_extension}。")
        
    docs = loader.load()
    full_text = "\n".join(doc.page_content for doc in docs)

    # 使用Chonky进行语义切分
    print("正在初始化 Chonky ParagraphSplitter...")
    splitter = ParagraphSplitter(device="cpu")
    chunks = list(splitter(full_text))
    print(f"文本切分完成，共 {len(chunks)} 个文本块。")

    # 转换为LangChain Document对象
    documents_for_db = [Document(page_content=chunk) for chunk in chunks]

    # 创建并保存FAISS数据库
    print(f"正在创建FAISS数据库...")
    db = FAISS.from_documents(documents_for_db, embedding_model)
    db.save_local(output_db_path)
    
    duration = time.time() - start_time
    print(f"✅ 向量数据库构建完成并保存至 '{output_db_path}'，耗时: {duration:.2f} 秒。")


# --- 2. 混合检索与生成的核心逻辑 ---

def run_hybrid_rag(llm_for_rag, embedding_model, query, kg_path, db_path, output_file):
    """执行完整的混合检索RAG流程并保存结果。"""

    # --- 定义内部检索函数 ---
    def load_knowledge_graph(path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def search_in_kg(q: str, kg_data: List[Dict]) -> str:
        entity_extraction_prompt = ChatPromptTemplate.from_template("从下面的问题中提取出所有的核心实体和关键词，以逗号分隔。问题: {question}")
        entity_extractor = entity_extraction_prompt | llm | StrOutputParser()
        entities = [e.strip() for e in entity_extractor.invoke({"question": q}).split(',')]
        print(f"从问题中提取的实体: {entities}")
        found_facts, graph_content = [], kg_data[0]
        for entity in entities:
            for node in graph_content.get("nodes", []):
                if entity.lower() in node.get("id", "").lower(): found_facts.append(f"找到实体节点: {node}")
            for rel in graph_content.get("relationships", []):
                if entity.lower() in rel.get("source", {}).get("id", "").lower() or entity.lower() in rel.get("target", {}).get("id", "").lower():
                    found_facts.append(f"找到相关关系: {rel}")
        return "从知识图谱中找到的精确事实：\n" + "\n".join(list(set(found_facts))) if found_facts else "在知识图谱中未找到直接相关的事实。"

    def retrieve_from_vectorstore(q: str, path: str, k: int = 3) -> str:
        print(f"正在从 '{path}' 加载向量数据库并检索...")
        db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        results = db.similarity_search(q, k=k)
        return "从相关文档中找到的上下文：\n" + "\n\n".join(doc.page_content for doc in results)
        
    # --- 编排与执行 ---
    knowledge_graph_data = load_knowledge_graph(kg_path)
    
    setup_and_retrieval = RunnableParallel(
        kg_context=lambda x: search_in_kg(x["question"], knowledge_graph_data),
        vector_context=lambda x: retrieve_from_vectorstore(x["question"], db_path),
        question=lambda x: x["question"],
    )

    final_prompt = ChatPromptTemplate.from_template("""
你是一个顶级的问答助手。请根据下面提供的“精确事实”和“相关上下文”，全面而准确地回答用户的问题。
在回答时，优先使用“精确事实”中的信息，并用“相关上下文”中的信息进行补充、解释和丰富。

用户问题: {question}

--- 上下文 ---
{kg_context}

{vector_context}
--- 结束 ---

最终答案:""")

    final_rag_chain = setup_and_retrieval | final_prompt | llm_for_rag | StrOutputParser()
    
    print(f"\n--- 开始执行混合检索 RAG ---")
    print(f"用户问题: {query}")
    answer = final_rag_chain.invoke({"question": query})
    
    print("\n\n--- 最终答案 ---")
    print(answer)

    # 将结果写入文件
    print(f"\n正在将答案写入文件: {output_file}")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"--- 用户问题 ---\n{query}\n\n--- 模型生成的答案 ---\n{answer}")
        print("✅ 答案写入成功。")
    except Exception as e:
        print(f"错误：写入文件失败: {e}")


# --- 3. 主执行入口：设置和解析命令行参数 ---

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(
        description="混合检索RAG系统的统一命令行工具",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息格式
    )
    
    parser.add_argument(
        "-i", "--input-file", 
        type=str, 
        default="input/KG_test.pdf",
        help="待处理的输入文件路径 (.pdf 或 .docx)。\n(默认: input/KG_test.pdf)"
    )
    parser.add_argument(
        "-q", "--query", 
        type=str, 
        default="What is the method of MastSAM?",
        help="需要提问的用户问题。\n(默认: 'What is the method of MastSAM?')"
    )
    parser.add_argument(
        "-o", "--output-file", 
        type=str, 
        default="rag_output/rag_answer.txt", 
        help="保存最终答案的文本文件路径。\n(默认: rag_output/rag_answer.txt)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true", # 动作参数，存在即为True
        help="强制重新构建知识图谱和向量数据库，即使它们已存在。"
    )

    args = parser.parse_args()
    
    # --- 智能文件路径管理 ---
    # 根据输入文件名，自动生成对应的KG和DB路径
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    kg_path = f"output/{base_filename}_KG.json"
    db_path = f"vector_db/{base_filename}_faiss_index"

    # --- 智能执行流程 ---
    # 1. 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 '{args.input_file}' 不存在。请检查路径或创建文件。")
        sys.exit(1)

    # 2. 检查是否需要强制重建或缓存不存在，如果需要，则运行构建步骤
    if args.force_rebuild:
        print("⚡️ 检测到 --force-rebuild 参数，将强制重建所有数据...")
        if os.path.exists(kg_path): os.remove(kg_path)
        if os.path.exists(db_path): shutil.rmtree(db_path) # 删除文件夹

    if not os.path.exists(kg_path):
        build_knowledge_graph(llm, args.input_file, kg_path)
    else:
        print(f"ℹ️ 知识图谱 '{kg_path}' 已存在，跳过构建步骤。")

    if not os.path.exists(db_path):
        build_vector_store(embedding_model, args.input_file, db_path)
    else:
        print(f"ℹ️ 向量数据库 '{db_path}' 已存在，跳过构建步骤。")

    # 3. 运行RAG问答
    run_hybrid_rag(llm, embedding_model, args.query, kg_path, db_path, args.output_file)
