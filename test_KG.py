import os
import json
import sys
import argparse

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from langchain_community.chat_models import ChatTongyi
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

from dotenv import load_dotenv

# 从环境变量中读取API Key
#========================================================================================
#========================================================================================
# =============       临时环境变量：windows,powershell                          ==========
# =============       输入$env:DASHSCOPE_API_KEY = "YOUR_DASHSCOPE_API_KEY"    ==========
# =============       更多请参考阿里云百炼-API参考-配置API Key到环境变量          ==========
#========================================================================================
#========================================================================================
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    print("错误：未找到'DASHSCOPE_API_KEY'环境变量。")
    sys.exit(1)

#运行指令：python test_KG.py --input_file input/your.pdf --output_file output/your_KG.json
if __name__ == "__main__":
    #===================================================================
    # ================在代码中直接设置输入文件路径和查询内容================
    #===================================================================
    parser = argparse.ArgumentParser(description="知识图谱抽取")
    parser.add_argument("--input_file", type=str, default="input/KG_test.pdf", help="输入文件路径（PDF或Word）")
    parser.add_argument("--output_file", type=str, default="output/test_KG.json", help="输出JSON文件路径")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    print("正在初始化Qwen模型...")
    try:
        llm = ChatTongyi(
            dashscope_api_key=DASHSCOPE_API_KEY,
            model_name="qwen-plus",
            temperature=0,
            max_tokens=10
        )
        llm.invoke("Hi") # 测试调用
        print("Qwen模型初始化并测试成功。")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        sys.exit(1)

    transformer = LLMGraphTransformer(llm=llm)

    #读取文档
    file_extension = os.path.splitext(input_file)[1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(input_file)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(input_file)
    else:
        raise ValueError(f"不支持的文件类型: {file_extension}。仅支持 .pdf 和 .docx。")

    print(f"正在加载文件: {input_file}...")
    text_content = loader.load()
    full_text = "\n".join(doc.page_content for doc in text_content)
    documents = [Document(page_content=full_text)]

    #调用大模型，生成知识图谱
    print("\n正在调用Qwen模型提取图谱信息...")
    graph_documents = transformer.convert_to_graph_documents(documents)

    output_folder = os.path.dirname(output_file)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    #将图谱信息写入JSON文件
    print(f"正在将图谱信息写入本地文件: {output_file}")
    output_data = []
    for doc in graph_documents:
        output_data.append({
            "nodes": [node.dict() for node in doc.nodes],
            "relationships": [rel.dict() for rel in doc.relationships],
            "source": doc.source.dict()
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("\n知识图谱已成功提取并保存到本地文件！")