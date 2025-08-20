import time
import os
import uuid
from typing import List, Tuple, Dict
from collections import Counter

# LangChain 和其他库的导入
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
import fitz
from chonky import ParagraphSplitter

'''
Hierarchical Indices（多级索引）

1.处理文档：pdf中的段落需要有标题，标题字号大于正文字号
2.切分文档：根据标题切分出父文档
3.切分子文档：根据段落切分出子文档
4.构建向量索引：使用子文档创建FAISS向量库
5.构建检索器：使用ParentDocumentRetriever将子文档与父文档关联
6.使用LLM路由：根据用户查询和标题，使用LLM决定查询属于哪个标题，并在该标题下进行检索

缺点：
1.锁定一个标题下的内容，可能错失其它标题下的相关信息
2.应用场景太狭隘，如果是论文的话为什么不直接用标题进行检索？还需要切出子文档再检索？
'''
# --- 1. PDF切分函数  ---
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
    print(f"分析完成：最常见的正文字号约为 {most_common_size}。将使用 > {most_common_size} 的字号作为标题识别阈值。")
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
                            parent_documents.append(
                                Document(
                                    page_content=current_content.strip(),
                                    metadata={"title": current_title, "source": file_path}
                                )
                            )
                        current_title = line_text
                        current_content = ""
                        print(f"  识别到新标题: '{current_title}' (第 {page_num + 1} 页)")
                    else:
                        current_content += line_text + "\n"
    if current_content.strip():
        parent_documents.append(
            Document(
                page_content=current_content.strip(),
                metadata={"title": current_title, "source": file_path}
            )
        )
    print(f"--- 智能切分完成，共生成 {len(parent_documents)} 个父文档（章节） ---")
    return parent_documents

# --- 2. Chonky 包装类 ---
class ChonkyTextSplitter(TextSplitter):
    def __init__(self, chonky_splitter: ParagraphSplitter, **kwargs):
        super().__init__(**kwargs)
        self.chonky_splitter = chonky_splitter
    def split_text(self, text: str) -> List[str]:
        return list(self.chonky_splitter(text))

# --- 3. 自定义检索函数  ---
def retrieve_with_title_preference(query: str, retriever, parent_store: InMemoryStore, k: int = 3) -> List[Document]:
    print("\n--- 开始执行标题优先的自定义检索 ---")
    all_parent_docs = parent_store.mget(list(parent_store.yield_keys()))
    all_titles = {doc.metadata.get('title', '').lower() for doc in all_parent_docs if doc.metadata.get('title')}
    print(f"可用的章节标题关键词: {all_titles}")
    matched_title = None
    query_lower = query.lower()
    for title in all_titles:
        if title in query_lower:
            matched_title = title
            print(f"在查询中识别到标题关键词: '{matched_title}'")
            break
    candidate_docs = retriever.invoke(query, k=max(k * 3, 10))
    print(f"初步检索到 {len(candidate_docs)} 个候选文档。")
    if not candidate_docs:
        print("警告：初步语义检索未返回任何候选文档。")
        return []
    if not matched_title:
        print("查询中未发现标题关键词，将返回标准语义检索结果。")
        return candidate_docs[:k]
    prioritized_docs = []; other_docs = []
    for doc in candidate_docs:
        doc_title = doc.metadata.get('title', '').lower()
        if doc_title == matched_title:
            prioritized_docs.append(doc)
        else:
            other_docs.append(doc)
    final_ranked_docs = prioritized_docs + other_docs
    seen_contents = set(); unique_docs = []
    for doc in final_ranked_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)
    print(f"重排完成，标题为 '{matched_title}' 的文档已被提升到最前。")
    return unique_docs[:k]

# --- 主执行流程  ---
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # --- 配置 ---
    pdf_file_path = "input/RAG_test_essay.pdf"
    os.makedirs("input", exist_ok=True)

    # --- 初始化 ---
    print("--- 初始化模型和切分器 ---")
    embedding_model = SentenceTransformerEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={}, encode_kwargs={'prompt_name': "query"}
    )
    child_splitter = ChonkyTextSplitter(chonky_splitter=ParagraphSplitter(device="cpu"))
    parent_doc_store = InMemoryStore()

    # --- 每次都重新创建索引和检索器，以保证ID同步 ---
    print("\n--- 开始构建父/子文档和向量索引 ---")
    
    # 1. 切分父文档
    parent_documents = split_pdf_by_headings(pdf_file_path)
    if not parent_documents:
        sys.exit("未能从PDF中切分出任何父文档，程序退出。")

    # 2. 准备父文档ID并存入store
    parent_doc_ids = [str(uuid.uuid4()) for _ in parent_documents]
    parent_doc_store.mset(list(zip(parent_doc_ids, parent_documents)))

    # 3. 切分子文档并关联父ID
    child_documents = []
    for i, doc in enumerate(parent_documents):
        _id = parent_doc_ids[i]
        sub_docs = child_splitter.split_documents([doc])
        for _doc in sub_docs:
            _doc.metadata["parent_id"] = _id
        child_documents.extend(sub_docs)

    # 保存父文档和子文档到 output/rag.txt
    with open("output/rag.txt", "w", encoding="utf-8") as f:
        f.write("=== 父文档章节 ===\n")
        for i, doc in enumerate(parent_documents):
            f.write(f"\n--- 父文档 {i+1} ---\n")
            f.write(f"标题: {doc.metadata.get('title', 'N/A')}\n")
            f.write(f"内容:\n{doc.page_content}\n")

        f.write("\n=== 子文档片段 ===\n")
        for i, doc in enumerate(child_documents):
            f.write(f"\n--- 子文档 {i+1} ---\n")
            f.write(f"父ID: {doc.metadata.get('parent_id', 'N/A')}\n")
            f.write(f"内容:\n{doc.page_content}\n")
    
    # 4. 使用稳健的方法创建FAISS向量库
    print(f"\n正在为 {len(child_documents)} 个子文档创建FAISS索引...")
    vectorstore = FAISS.from_documents(child_documents, embedding_model)
    
    # 5. 初始化可以工作的检索器
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_doc_store,
        child_splitter=child_splitter # 确保传入
    )
    print("--- 索引和检索器构建完成 ---")

    # --- 执行检索 ---
    print("\n" + "="*20 + " 开始检索 " + "="*20)
    user_query = os.environ.get("USER_QUERY_PDF", "Explain the method of MastSAM").replace("_", " ")
    print(f"查询: '{user_query}'")
    
    start_time = time.time()
    retrieved_docs = retrieve_with_title_preference(
        query=user_query,
        retriever=retriever,
        parent_store=parent_doc_store,
        k=3
    )

    end_time = time.time()
    
    print(f"\n自定义检索完成。总耗时: {end_time - start_time:.4f} 秒")
    print("--- 最终检索结果 ---")
    if not retrieved_docs:
        print("未能找到任何相关文档。")
    else:
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- [相关父文档片段 {i+1} (已重排)] ---")
            print(f"来源章节标题: '{doc.metadata.get('title', 'N/A')}'")
            print(f"内容:\n{doc.page_content}\n")


           