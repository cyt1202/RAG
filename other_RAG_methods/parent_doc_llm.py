import time
import os
import uuid
from typing import List, Tuple, Dict
from collections import Counter

# Qwen API
from openai import OpenAI

# LangChain 和其他库的导入
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
import fitz  # PyMuPDF
from chonky import ParagraphSplitter

'''
在parent_document_retriever.py的基础上接入llm
用llm决定query属于哪个标题，在改标题下文本块进行检索'''

# --- 1. Qwen API 路由函数 (保持不变) ---
def get_most_relevant_section_by_qwen(query: str, titles: set, client: OpenAI) -> str:
    print("\n--- 调用Qwen-plus进行智能路由决策 ---")
    titles_str = "\n".join(f"- {title.capitalize()}" for title in sorted(list(titles)))
    system_content = "你是一个专业的AI研究助理。你的任务是根据用户的提问，从一篇论文的章节目录中，选择最可能包含答案的那个章节。你的回答必须是下面目录中的一个，或者回答 'none' 表示没有合适的章节。不要添加任何解释或多余的文字。"
    user_content = f"""
# 可选章节目录:
{titles_str}

# 用户提问:
"{query}"

请直接输出你选择的最相关章节标题（小写）。
"""
    try:
        completion = client.chat.completions.create(model="qwen-plus", messages=[{"role": "system", "content": system_content}, {"role": "user", "content": user_content}], temperature=0)
        predicted_title = completion.choices[0].message.content.strip().lower()
        print(f"Qwen决策结果: '{predicted_title}'")
        return predicted_title if predicted_title in titles else "none"
    except Exception as e:
        print(f"错误：调用Qwen API失败: {e}"); return "none"

# --- 2. ⭐ 全新的混合检索函数 ---
def hybrid_retrieval_with_llm_router(
    query: str,
    parent_store: InMemoryStore,
    main_vectorstore: FAISS,
    qwen_client: OpenAI,
    embedding_model,
    k_target: int = 3,
    k_global: int = 3
) -> List[Document]:
    print("\n--- 开始执行LLM路由 + 混合检索 ---")
    all_parent_docs_keys = list(parent_store.yield_keys())
    all_parent_docs_values = parent_store.mget(all_parent_docs_keys)
    all_titles = {doc.metadata.get('title', '').lower() for doc in all_parent_docs_values if doc and doc.metadata.get('title')}
    
    predicted_title = get_most_relevant_section_by_qwen(query, all_titles, qwen_client)
    
    targeted_results = []
    if predicted_title != "none":
        print(f"LLM已锁定目标章节: '{predicted_title}'。开始执行靶向搜索...")
        target_parent_ids = {doc_id for doc_id, doc in zip(all_parent_docs_keys, all_parent_docs_values) if doc and doc.metadata.get('title', '').lower() == predicted_title}
        
        all_child_docs = list(main_vectorstore.docstore._dict.values())
        target_child_docs = [doc for doc in all_child_docs if doc.metadata.get("parent_id") in target_parent_ids]

        if target_child_docs:
            try:
                targeted_vectorstore = FAISS.from_documents(target_child_docs, embedding_model)
                targeted_results = targeted_vectorstore.similarity_search(query, k=k_target)
                print(f"在目标章节 '{predicted_title}' 中找到 {len(targeted_results)} 个相关子文档。")
            except Exception as e:
                print(f"错误：靶向搜索失败: {e}。跳过靶向搜索。")
        else:
            print("警告：在向量库中未找到属于该目标章节的子文档。")

    print("\n正在执行全局补充搜索...")
    global_results = main_vectorstore.similarity_search(query, k=k_target + k_global)
    print(f"在全局范围中找到 {len(global_results)} 个相关子文档。")

    final_docs = []
    seen_contents = set()

    for doc in targeted_results:
        if doc.page_content not in seen_contents:
            final_docs.append(doc)
            seen_contents.add(doc.page_content)
    
    for doc in global_results:
        if len(final_docs) >= k_target + k_global:
            break
        if doc.page_content not in seen_contents:
            final_docs.append(doc)
            seen_contents.add(doc.page_content)
            
    print(f"\n合并去重后，最终返回 {len(final_docs)} 个最相关的子文档。")
    return final_docs

# --- 3. PDF处理和Chonky包装类 (保持不变) ---
# ... (此处省略 split_pdf_by_headings 和 ChonkyTextSplitter 的代码, 请确保它们存在)
def get_font_size_statistics(doc: fitz.Document) -> float:
    font_counts = Counter()
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        font_size = round(s["size"])
                        font_counts[font_size] += 1
    if not font_counts: return 12.0
    most_common_size = font_counts.most_common(1)[0][0]
    title_font_size_threshold = most_common_size + 1.0
    print(f"分析完成：最常见的正文字号约为 {most_common_size}。")
    return title_font_size_threshold

def split_pdf_by_headings(file_path: str) -> List[Document]:
    print(f"\n--- 开始基于标题智能切分PDF: {file_path} ---")
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"错误: 打开或解析PDF文件 {file_path} 失败: {e}")
        return []
    title_font_size_threshold = get_font_size_statistics(doc)
    parent_documents = []; current_content = ""; current_title = "Introduction"
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    line_text = "".join(s["text"] for s in l["spans"]).strip()
                    if not line_text: continue
                    main_font_size = round(l["spans"][0]["size"])
                    if main_font_size > title_font_size_threshold and len(line_text) < 100:
                        if current_content.strip():
                            parent_documents.append(Document(page_content=current_content.strip(), metadata={"title": current_title, "source": file_path}))
                        current_title = line_text
                        current_content = ""
                        print(f"  识别到新标题: '{current_title}' (第 {page_num + 1} 页)")
                    else:
                        current_content += line_text + "\n"
    if current_content.strip():
        parent_documents.append(Document(page_content=current_content.strip(), metadata={"title": current_title, "source": file_path}))
    print(f"--- 智能切分完成，共生成 {len(parent_documents)} 个父文档（章节） ---")
    return parent_documents

class ChonkyTextSplitter(TextSplitter):
    def __init__(self, chonky_splitter: ParagraphSplitter, **kwargs):
        super().__init__(**kwargs)
        self.chonky_splitter = chonky_splitter
    def split_text(self, text: str) -> List[str]:
        return list(self.chonky_splitter(text))

# --- 主执行流程 ---
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # --- 配置 ---
    pdf_file_path = "input/RAG_test_essay.pdf"
    os.makedirs("input", exist_ok=True)

    # --- 初始化所有模型和客户端 ---
    print("--- 初始化模型和客户端 ---")
    try:
        qwen_client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        qwen_client.chat.completions.create(model="qwen-plus", messages=[{"role":"user", "content":"hi"}], max_tokens=10)
        print("Qwen API客户端初始化并测试成功。")
    except Exception as e:
        sys.exit(f"错误：初始化或测试Qwen API客户端失败: {e}")

    embedding_model = SentenceTransformerEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={}, encode_kwargs={'prompt_name': "query"})
    child_splitter = ChonkyTextSplitter(chonky_splitter=ParagraphSplitter(device="cpu"))
    parent_doc_store = InMemoryStore()

    # --- 每次都重新创建索引 ---
    print("\n--- 开始构建父/子文档和向量索引 ---")
    parent_documents = split_pdf_by_headings(pdf_file_path)
    if not parent_documents: sys.exit("未能从PDF中切分出任何父文档。")

    parent_doc_ids = [str(uuid.uuid4()) for _ in parent_documents]
    parent_doc_store.mset(list(zip(parent_doc_ids, parent_documents)))

    child_documents = []
    for i, doc in enumerate(parent_documents):
        _id = parent_doc_ids[i]
        sub_docs = child_splitter.split_documents([doc])
        for _doc in sub_docs: _doc.metadata["parent_id"] = _id
        child_documents.extend(sub_docs)

    print(f"\n正在为 {len(child_documents)} 个子文档创建FAISS索引...")
    main_vectorstore = FAISS.from_documents(child_documents, embedding_model)
    print("--- 索引构建完成 ---")

    # --- ⭐ 执行新的混合检索 ---
    print("\n" + "="*20 + " 开始检索 " + "="*20)
    user_query = os.environ.get("USER_QUERY_PDF", "Explain the method of MastSAM").replace("_", " ")
    print(f"查询: '{user_query}'")

    start_time = time.time()
    retrieved_docs = hybrid_retrieval_with_llm_router(
        query=user_query,
        parent_store=parent_doc_store,
        main_vectorstore=main_vectorstore,
        qwen_client=qwen_client,
        embedding_model=embedding_model,
        k_target=3,
        k_global=3
    )
    end_time = time.time()

    print(f"\n混合检索完成。总耗时: {end_time - start_time:.4f} 秒")
    print("--- 最终检索结果 ---")
    if not retrieved_docs:
        print("未能找到任何相关文档。")
    else:
        for i, doc in enumerate(retrieved_docs):
            # 现在打印的是子文档的信息
            print(f"\n--- [相关子文档 {i+1}] ---")
            print(f"来源章节标题: '{doc.metadata.get('title', 'N/A')}'")
            print(f"内容:\n{doc.page_content}\n")
