# -*- coding: utf-8 -*-
"""
Markdown 文档分块处理脚本

功能：读取指定目录下的 Markdown 文档，根据标题层级进行分块处理，
移除图片内容，将每块转换为 JSON 格式存储，供 RAG（检索增强生成）系统使用。

分块策略：
- 优先使用三级标题（###）进行分块
- 若无三级标题，则使用二级标题（##）分块
- 若无二级标题，则使用一级标题（#）分块

输出格式：
[{
    "content": "内容",
    "title_one": "一级标题名称",
    "title_two": "二级标题名称",
    "title_three": "三级标题名称",
    "source_file": "源文件名称",
    "source_path": "源文件绝对路径",
    "chunk_id": "唯一标识符"
}]
"""

import os
import re
import json
import uuid
from pathlib import Path


def remove_images(content):
    """
    移除 Markdown 内容中的图片引用
    
    支持两种格式：
    1. 标准 Markdown 图片语法：![alt text](image_path)
    2. Obsidian 风格图片语法：![[image_path]]
    
    Args:
        content: 原始 Markdown 内容字符串
        
    Returns:
        移除图片后的纯文本内容
    """
    # 移除标准 Markdown 图片语法：![alt](path)
    content = re.sub(r"!\[.*?\]\([^)]+\)", "", content)
    # 移除 Obsidian 风格图片语法：![[path]]
    content = re.sub(r"!\[\[.*?\]\]", "", content)
    return content


def detect_chunk_level(lines):
    """
    根据文件行列表检测文档的分块级别
    
    按优先级检测文档中存在的标题级别：
    1. 若存在三级标题（###），返回 3
    2. 若存在二级标题（##），返回 2
    3. 若存在一级标题（#），返回 1
    4. 若无标题，返回 0
    
    注意：代码块（```）内的 # 注释不会被误识别为标题
    
    Args:
        lines: 文件内容的行列表（已按换行符分割）
        
    Returns:
        int: 分块级别（0-3）
    """
    has_h3 = False  # 是否存在三级标题
    has_h2 = False  # 是否存在二级标题
    has_h1 = False  # 是否存在一级标题
    in_code_block = False  # 是否在代码块内
    
    for line in lines:
        stripped = line.strip()
        
        # 检测代码块边界，切换状态
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        
        # 跳过代码块内的内容，避免将 Python 注释 # 误识别为标题
        if in_code_block:
            continue
        
        # 检测三级标题（排除四级及以上）
        if stripped.startswith("### ") and not stripped.startswith("####"):
            has_h3 = True
        # 检测二级标题（排除三级及以上）
        elif stripped.startswith("## ") and not stripped.startswith("###"):
            has_h2 = True
        # 检测一级标题（排除二级及以上）
        elif stripped.startswith("# ") and not stripped.startswith("##"):
            has_h1 = True
        
        # 一旦检测到三级标题，提前退出循环
        if has_h3:
            break
    
    # 按优先级返回分块级别
    if has_h3:
        return 3
    elif has_h2:
        return 2
    elif has_h1:
        return 1
    return 0


def parse_markdown(lines, file_path, chunk_level):
    """
    解析 Markdown 文件，按指定级别分块
    
    遍历文件的每一行，维护当前标题上下文（一级、二级、三级），
    当遇到新的同级或更高级标题时，将当前累积的内容生成为一个 chunk。
    
    改进：确保所有内容都能被保存，不会出现数据丢失：
    - 当 chunk_level==3 时，若内容在 H1/H2 下但无 H3，仍会保存为 chunk
    - 当 chunk_level==2 时，若内容在 H1 下但无 H2，仍会保存为 chunk
    
    Args:
        lines: 文件内容的行列表（已按换行符分割）
        file_path: Markdown 文件的绝对路径（用于生成 source_file 和 source_path）
        chunk_level: 分块级别（1/2/3）
        
    Returns:
        list: chunk 对象列表，每个对象包含 content、title_one、title_two、
              title_three、source_file、source_path、chunk_id 字段
    """
    chunks = []  # 存储生成的所有 chunk
    
    # 当前标题上下文
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_content = []  # 当前累积的内容行
    in_code_block = False  # 是否在代码块内
    
    # 源文件信息
    source_file = os.path.basename(file_path)
    source_path = str(Path(file_path).resolve())
    
    def add_chunk():
        """
        将当前累积的内容生成为一个 chunk
        
        内部函数，负责：
        1. 检查当前内容是否为空
        2. 移除图片并清理空白
        3. 根据分块级别和当前标题上下文生成 chunk（确保无数据丢失）
        4. 清空当前内容累积器
        """
        nonlocal current_content, current_h1, current_h2, current_h3, chunks
        
        # 内容为空则跳过
        if not current_content:
            return
        
        # 移除图片并清理首尾空白
        cleaned_content = remove_images("\n".join(current_content)).strip()
        
        # 移除图片后内容为空则跳过
        if not cleaned_content:
            return
        
        # 根据分块级别生成不同类型的 chunk
        if chunk_level == 3:
            # 三级标题分块：优先使用三级标题，其次二级，最后一级
            # 确保内容不会丢失：只要有任一标题存在就保存
            if current_h3:
                chunks.append({
                    "content": cleaned_content,
                    "title_one": current_h1,
                    "title_two": current_h2,
                    "title_three": current_h3,
                    "source_file": source_file,
                    "source_path": source_path,
                    "chunk_id": str(uuid.uuid4())  # 生成唯一标识符
                })
            elif current_h2:
                # 无三级标题时，使用二级标题
                chunks.append({
                    "content": cleaned_content,
                    "title_one": current_h1,
                    "title_two": current_h2,
                    "title_three": "",
                    "source_file": source_file,
                    "source_path": source_path,
                    "chunk_id": str(uuid.uuid4())
                })
            elif current_h1:
                # 无二级标题时，使用一级标题（防止数据丢失）
                chunks.append({
                    "content": cleaned_content,
                    "title_one": current_h1,
                    "title_two": "",
                    "title_three": "",
                    "source_file": source_file,
                    "source_path": source_path,
                    "chunk_id": str(uuid.uuid4())
                })
        elif chunk_level == 2:
            # 二级标题分块：优先使用二级标题，其次一级（防止数据丢失）
            if current_h2:
                chunks.append({
                    "content": cleaned_content,
                    "title_one": current_h1,
                    "title_two": current_h2,
                    "title_three": "",
                    "source_file": source_file,
                    "source_path": source_path,
                    "chunk_id": str(uuid.uuid4())
                })
            elif current_h1:
                # 无二级标题时，使用一级标题（防止数据丢失）
                chunks.append({
                    "content": cleaned_content,
                    "title_one": current_h1,
                    "title_two": "",
                    "title_three": "",
                    "source_file": source_file,
                    "source_path": source_path,
                    "chunk_id": str(uuid.uuid4())
                })
        elif chunk_level == 1:
            # 一级标题分块
            if current_h1:
                chunks.append({
                    "content": cleaned_content,
                    "title_one": current_h1,
                    "title_two": current_h2,
                    "title_three": "",
                    "source_file": source_file,
                    "source_path": source_path,
                    "chunk_id": str(uuid.uuid4())
                })
        
        # 清空当前内容累积器
        current_content = []
    
    # 逐行解析 Markdown 文件
    for line in lines:
        stripped = line.strip()
        
        # 检测代码块边界
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current_content.append(line)
            continue
        
        # 代码块内的内容直接加入，不进行标题检测
        if in_code_block:
            current_content.append(line)
            continue
        
        # 检测一级标题（#）
        if stripped.startswith("# ") and not stripped.startswith("##"):
            add_chunk()  # 先保存当前累积的内容
            current_h1 = stripped[2:]  # 提取标题文本（去掉 "# "）
            current_h2 = ""  # 重置二级和三级标题
            current_h3 = ""
        
        # 检测二级标题（##）
        elif stripped.startswith("## ") and not stripped.startswith("###"):
            add_chunk()  # 先保存当前累积的内容
            current_h2 = stripped[3:]  # 提取标题文本（去掉 "## "）
            current_h3 = ""  # 重置三级标题
        
        # 检测三级标题（###）
        elif stripped.startswith("### ") and not stripped.startswith("####"):
            # 只有在三级分块级别时，才在遇到三级标题时分块
            if chunk_level == 3:
                add_chunk()  # 先保存当前累积的内容
            current_h3 = stripped[4:]  # 提取标题文本（去掉 "### "）
        
        # 普通内容行
        else:
            current_content.append(line)
    
    # 处理文件末尾剩余的内容
    add_chunk()
    
    return chunks


def main():
    """
    主函数：处理所有 Markdown 文件
    
    流程：
    1. 创建输出目录
    2. 遍历源目录下所有 .md 文件
    3. 对每个文件读取一次，检测分块级别并进行分块处理（避免重复 I/O）
    4. 将结果写入 JSON 文件（文件名与源文件对应）
    """
    # 配置路径
    source_dir = r"d:\code\ai-study\week10\data\source"  # 源 Markdown 文件目录
    chunks_dir = r"d:\code\ai-study\week10\data\chunks"  # 输出 JSON 文件目录
    
    # 确保输出目录存在
    os.makedirs(chunks_dir, exist_ok=True)
    
    # 获取所有 Markdown 文件并排序
    md_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".md")])
    
    # 逐个处理文件
    for md_file in md_files:
        file_path = os.path.join(source_dir, md_file)
        
        # 读取文件一次，供后续检测和解析使用（避免重复 I/O）
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        
        # 检测分块级别（传入已读取的行列表）
        chunk_level = detect_chunk_level(lines)
        
        # 解析文件并分块（传入已读取的行列表）
        chunks = parse_markdown(lines, file_path, chunk_level)
        
        # 生成输出文件名（.md -> .json）
        json_file = os.path.join(chunks_dir, md_file.replace(".md", ".json"))
        
        # 写入 JSON 文件
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # 打印处理结果
        print(f"Processed: {md_file} -> {len(chunks)} chunks")


if __name__ == "__main__":
    main()