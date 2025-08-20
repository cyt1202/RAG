@echo off
setlocal enabledelayedexpansion

REM 这是一个批处理脚本，用于测试不同的文本分割方法并记录实验结果。
REM 最终选择chonky，因为它的文档分割的语义信息保留最完整

REM 设置你的查询
set USER_QUERY_PDF=Explain the method of MastSAM
set USER_QUERY_DOCX=What is Chronomirror

REM 运行 Chonky
echo Running tokenlization_chonky.py ...
echo [Chonky Splitter] >> experiment.txt
echo Query(PDF): %USER_QUERY_PDF% >> experiment.txt
echo Query(Word): %USER_QUERY_DOCX% >> experiment.txt
echo Model: Qwen/Qwen3-Embedding-0.6B >> experiment.txt
python tokenlization_chonky.py >> experiment.txt

REM 运行 Splitter
echo. >> experiment.txt
echo [Sentence Splitter] >> experiment.txt
echo Query(PDF): %USER_QUERY_PDF% >> experiment.txt
echo Query(Word): %USER_QUERY_DOCX% >> experiment.txt
echo Model: Qwen/Qwen3-Embedding-0.6B >> experiment.txt
python tokenlization_splitter.py >> experiment.txt

REM 运行 Easy
echo. >> experiment.txt
echo [Easy Splitter] >> experiment.txt
echo Query(PDF): %USER_QUERY_PDF% >> experiment.txt
echo Query(Word): %USER_QUERY_DOCX% >> experiment.txt
echo Model: Qwen/Qwen3-Embedding-0.6B >> experiment.txt
python tokenlization_easy.py >> experiment.txt

echo.
echo ==== 实验全部完成，结果已保存到 experiment.txt ====
pause