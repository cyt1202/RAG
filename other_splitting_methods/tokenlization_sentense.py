import time
import os
import numpy as np
from sentence_splitter import SentenceSplitter
from typing import List

# 导入 LangChain 文档加载器
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

'''
自定义的文本分割器：
在基于分隔符的递归切分的基础上，增加了句子分割逻辑。
确保切分点总是在句子末尾。
'''
class CustomSentenceSplitter:
    """
    一个自定义的文本分割器，它在段落和空格分割之间加入了句子分割逻辑。
    确保切分点总是在句子末尾。
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化分割器。

        参数:
            chunk_size (int): 每个文本块的目标最大字符数。
            chunk_overlap (int): 文本块之间的重叠字符数。
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 关键：确保使用中文语言包
        self.sentence_splitter = SentenceSplitter(language='en')

    def split(self, text: str) -> List[str]:
        """对输入的长文本进行切分。"""
        if not text:
            return []

        # 1. 初步按段落分割，并清理空白
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # 2. 将所有段落切分成一个完整的句子列表
        all_sentences = []
        for p in paragraphs:
            sentences = self.sentence_splitter.split(p)
            all_sentences.extend(sentences)
        
        if not all_sentences:
            return []

        # 3. 将句子组合成块 (chunks)
        chunks = []
        current_chunk_sentences = []
        current_chunk_len = 0

        for sentence in all_sentences:
            sentence_len = len(sentence)
            # 如果加入新句子会超长，则先将当前块保存
            if current_chunk_len + sentence_len + 1 > self.chunk_size and current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                
                # --- 处理重叠 (Overlap) ---
                # 从后向前找到可以作为重叠部分的句子
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                
                # 开始新的一个块
                current_chunk_sentences = overlap_sentences
                current_chunk_len = overlap_len
            
            # 将当前句子加入块中
            current_chunk_sentences.append(sentence)
            current_chunk_len += sentence_len + 1 # +1 for the space

        # 添加最后一个块
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks

def process_file(splitter: CustomSentenceSplitter, file_path: str, output_path: str):
    """加载、处理单个文件并保存结果"""
    print(f"\n--- 开始处理文件: {file_path} ---")
    start_time = time.time()

    # 1. 加载文档内容
    full_text = ""
    if not os.path.exists(file_path):
        print(f"错误：文件未找到: {file_path}")
        return

    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")

        documents = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in documents])

    except Exception as e:
        print(f"错误：加载文件 {file_path} 失败: {e}")
        return

    print(f" - 文件加载成功，总字数: {len(full_text)}")

    # 2. 执行自定义的句子感知切分
    chunks = splitter.split(full_text)
    
    # 3. 写入输出文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"源文件: {file_path}\n")
            f.write(f"自定义句子切分结果 (共 {len(chunks)} 个块):\n")
            f.write("="*40 + "\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} (长度: {len(chunk)}) ---\n")
                f.write(chunk.strip())
                f.write("\n\n" + "="*40 + "\n\n")
        print(f" - 切分结果已成功写入到: {output_path}")
    except Exception as e:
        print(f"错误：写入文件 {output_path} 失败: {e}")

    # 4. 记录并打印总耗时
    end_time = time.time()
    duration = end_time - start_time
    print(f"--- 处理 {file_path} 总耗时: {duration:.2f} 秒 ---")


if __name__ == "__main__":
    
    # 初始化我们新的自定义分割器
    # 您可以调整 chunk_size 和 chunk_overlap
    custom_splitter = CustomSentenceSplitter(chunk_size=300, chunk_overlap=40)

    # 定义要处理的文件和对应的输出路径
    # 请确保您有 sample.pdf 和 sample.docx 文件在同一目录下
    files_to_process = {
        "input/sample.pdf": "output/Sentence_chunk_pdf.txt",
        "input/sample.docx": "output/Sentence_chunk_word.txt"
    }

    # 循环处理所有文件
    for input_file, output_file in files_to_process.items():
        process_file(custom_splitter, input_file, output_file)