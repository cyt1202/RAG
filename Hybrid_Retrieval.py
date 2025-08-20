import os
import json
import sys
import argparse # 【新增】导入argparse库
from typing import List, Dict

# LangChain 相关导入
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document

# --- 0. 环境和模型初始化 ---
#========================================================================================
#========================================================================================
#==============       临时环境变量：windows,powershell                          ==========
#==============       输入$env:DASHSCOPE_API_KEY = "YOUR_DASHSCOPE_API_KEY"    ==========
#==============       检查环境是否生效：echo $env:DASHSCOPE_API_KEY             ========== 
#==============       更多请参考阿里云百炼-API参考-配置API Key到环境变量          ==========
#========================================================================================
#========================================================================================
from dotenv import load_dotenv
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误：未找到'DASHSCOPE_API_KEY'环境变量。")
    sys.exit(1)

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    max_tokens=2048,
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
print("答案生成LLM初始化成功。")

embedding_model = SentenceTransformerEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={},
    encode_kwargs={'prompt_name': "query"}
)
print("嵌入模型初始化成功。")

# --- 1. 定义检索函数 ---
def load_knowledge_graph(kg_path: str) -> Dict:
    print(f"正在从 '{kg_path}' 加载知识图谱...")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"知识图谱文件未找到: {kg_path}。请先运行 `1_build_kg.py`。")
    with open(kg_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def search_in_kg(query: str, kg_data: List[Dict]) -> str:
    entity_extraction_prompt = ChatPromptTemplate.from_template("从下面的问题中提取出所有的核心实体和关键词，以逗号分隔。问题: {question}")
    entity_extractor = entity_extraction_prompt | llm | StrOutputParser()
    entities = [e.strip() for e in entity_extractor.invoke({"question": query}).split(',')]
    
    print(f"从问题中提取的实体: {entities}")
    found_facts = []
    # 假设kg_data是一个列表，其第一个元素包含nodes和relationships
    if not kg_data:
        return "知识图谱数据为空。"
    graph_content = kg_data[0]
    
    for entity in entities:
        for node in graph_content.get("nodes", []):
            if entity.lower() in node.get("id", "").lower():
                found_facts.append(f"找到实体节点: {node}")
        for rel in graph_content.get("relationships", []):
            source_match = entity.lower() in rel.get("source", {}).get("id", "").lower()
            target_match = entity.lower() in rel.get("target", {}).get("id", "").lower()
            if source_match or target_match:
                found_facts.append(f"找到相关关系: {rel}")

    if not found_facts:
        return "在知识图谱中未找到直接相关的事实。"
    return "从知识图谱中找到的精确事实：\n" + "\n".join(list(set(found_facts)))

def retrieve_from_vectorstore(query: str, db_path: str, k: int = 3) -> str:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"向量数据库未找到: {db_path}。请先运行 `2_build_vectorstore.py`。")
    
    print(f"正在从 '{db_path}' 加载向量数据库并检索...")
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=k)
    return "从相关文档中找到的上下文：\n" + "\n\n".join(doc.page_content for doc in results)

# --- 2. 将核心逻辑封装到 main 函数中 ---
def main(args):
    """主执行函数，接收命令行参数"""
    
    # 从参数获取输入路径和问题
    kg_file_path = args.kg_path
    pdf_db_path = args.db_path
    user_query = args.query
    output_file_path = args.output_file

    # 加载知识图谱
    knowledge_graph_data = load_knowledge_graph(kg_file_path)

    # 编排混合检索流程
    setup_and_retrieval = RunnableParallel(
        kg_context=lambda x: search_in_kg(x["question"], knowledge_graph_data),
        vector_context=lambda x: retrieve_from_vectorstore(x["question"], pdf_db_path),
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

    最终答案:
    """)

    final_rag_chain = setup_and_retrieval | final_prompt | llm | StrOutputParser()

    # 执行RAG流程
    print(f"\n--- 开始执行混合检索 RAG ---")
    print(f"用户问题: {user_query}")
    
    answer = final_rag_chain.invoke({"question": user_query})
    
    print("\n\n--- 最终答案 ---")
    print(answer)

    # 【新增】将结果写入文件
    print(f"\n正在将答案写入文件: {output_file_path}")
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"--- 用户问题 ---\n")
            f.write(f"{user_query}\n\n")
            f.write(f"--- 模型生成的答案 ---\n")
            f.write(answer)
        print("答案写入成功。")
    except Exception as e:
        print(f"错误：写入文件失败: {e}")


# --- 3. 【新增】主执行入口：设置和解析命令行参数 ---
#运行指令：python Hybrid_Retrieval.py "What is the method of MastSAM?" --kg-path "my_data/kg.json" --db-path "my_data/faiss_db" --output-file "results/answer1.txt"
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')

    # 创建一个解析器
    parser = argparse.ArgumentParser(description="混合检索RAG系统的命令行接口")

    # 添加参数
    #===================================================================
    # ================在代码中直接设置输入文件路径和查询内容================
    #===================================================================
    parser.add_argument(
        "query", 
        type=str, 
        default="What is MastSAM?",
        help="需要提问的用户问题 (必填)"
    )
    parser.add_argument(
        "--kg-path", 
        type=str, 
        default="output/test_KG.json", 
        help="知识图谱JSON文件的路径 (默认: output/test_KG.json)"
    )
    parser.add_argument(
        "--db-path", 
        type=str, 
        default="vector_db/faiss_pdf_chonky_index", 
        help="FAISS向量数据库的文件夹路径 (默认: vector_db/faiss_pdf_chonky_index)"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="output/rag_answer.txt", 
        help="保存最终答案的文本文件路径 (默认: rag_answer/rag_answer.txt)"
    )

    # 解析参数
    args = parser.parse_args()
    
    # 调用主函数
    main(args)