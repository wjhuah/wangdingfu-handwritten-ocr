"""
适配2-79图片编号规则：
- 触发规则：空行 + 数字单独成行（数字=图片号，如2→0002.jpg）
- 文本块：数字行后 → 下一个数字行前（每个数字对应1张图片的文本）
- 生成：带标点/无标点文本块 + 图片-文本映射文件
"""
import os
import re
from typing import Dict, List

# ========== 核心配置（无需修改） ==========
START_NUM = 2    # 起始图片编号
END_NUM = 79     # 结束图片编号
PUNCT_PATTERN = re.compile(  # 全覆盖标点正则（彻底移除）
    r'[，。！？；：""''""''（）()【】\[\]、—…·《》<>「」『』｛｝￥＄％％＆＆＊＊＋＋－－／／＼＼｜｜～～｀｀､､·。]'
    r'|[,!?:;"\'(){}<>%&*+-/\\|~`.$]'
)

def split_text_by_image_num(raw_text_path: str) -> Dict[int, List[str]]:
    """
    按图片编号拆分文本块（核心逻辑）：
    1. 识别空行+数字单独行 → 标记图片号
    2. 提取数字行后到下一个数字行前的所有文本行
    3. 返回：{图片号: 文本行列表}
    """
    if not os.path.exists(raw_text_path):
        raise FileNotFoundError(f"原始文本文件不存在：{raw_text_path}")
    
    # 读取原始文本（保留换行，便于识别空行+数字行）
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]  # 保留换行符但去除末尾\n
    
    # 初始化变量
    image_text_map = {}  # {图片号: 文本行列表}
    current_num = None   # 当前匹配的图片号
    current_lines = []   # 当前图片的文本行
    
    for line in lines:
        line_stripped = line.strip()
        
        # 1. 识别「数字单独成行」（空行后+纯数字行）
        if line_stripped.isdigit():
            num = int(line_stripped)
            # 仅处理2-79的编号
            if START_NUM <= num <= END_NUM:
                # 先保存上一个图片的文本（如果有）
                if current_num is not None and current_lines:
                    image_text_map[current_num] = current_lines
                # 重置：开始新图片的文本收集
                current_num = num
                current_lines = []
            continue
        
        # 2. 收集当前图片的文本行（跳过空行）
        if current_num is not None and line_stripped:
            current_lines.append(line)
    
    # 保存最后一个图片的文本
    if current_num is not None and current_lines:
        image_text_map[current_num] = current_lines
    
    # 验证：确保2-79编号全覆盖
    missing_nums = [num for num in range(START_NUM, END_NUM+1) if num not in image_text_map]
    if missing_nums:
        print(f"⚠️  警告：以下图片编号未匹配到文本 → {missing_nums}")
    else:
        print(f"✅ 成功匹配所有编号：{START_NUM}-{END_NUM}（共{len(image_text_map)}个文本块）")
    
    return image_text_map

def clean_text_blocks(
    image_text_map: Dict[int, List[str]],
    output_dir: str,
    keep_punctuation: bool = True
) -> Dict[int, List[str]]:
    """
    清洗文本块：
    - 保留标点：仅去除首尾空白（端到端模型）
    - 移除标点：彻底移除所有标点+首尾空白（纯OCR模型）
    """
    cleaned_map = {}
    for num, lines in image_text_map.items():
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            # 移除标点（仅无标点版本）
            if not keep_punctuation:
                line_stripped = PUNCT_PATTERN.sub('', line_stripped)
            if line_stripped:  # 跳过空行
                cleaned_lines.append(line_stripped)
        cleaned_map[num] = cleaned_lines
    # 保存清洗后的文本文件
    suffix = "with_punct" if keep_punctuation else "no_punct"
    output_path = os.path.join(output_dir, f"cleaned_text_blocks_{suffix}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for num in sorted(cleaned_map.keys()):
            f.write(f"===== 图片编号：{num}（000{num}.jpg）=====\n")
            f.write('\n'.join(cleaned_map[num]) + '\n\n')
    # 保存映射文件（方便核对）
    map_path = os.path.join(output_dir, f"image_text_map_{suffix}.txt")
    with open(map_path, 'w', encoding='utf-8') as f:
        for num in sorted(cleaned_map.keys()):
            f.write(f"图片000{num}.jpg → 文本行数：{len(cleaned_map[num])}\n")
    print(f"✅ 清洗完成（{'保留标点' if keep_punctuation else '移除标点'}）→ {output_path}")
    return cleaned_map

def main(raw_text_path: str, output_dir: str):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1：按图片编号拆分文本块
    print("===== 第一步：按图片编号拆分文本块 =====")
    image_text_map = split_text_by_image_num(raw_text_path)
    
    # 步骤2：生成带标点版本（端到端模型）
    print("\n===== 第二步：生成带标点文本块 =====")
    clean_text_blocks(image_text_map, output_dir, keep_punctuation=True)
    
    # 步骤3：生成无标点版本（纯OCR模型）
    print("\n===== 第三步：生成无标点文本块 =====")
    clean_text_blocks(image_text_map, output_dir, keep_punctuation=False)

if __name__ == "__main__":
    # 路径配置（和你的项目结构对齐）
    RAW_TEXT_PATH = "./data/raw_text_original.txt"
    OUTPUT_DIR = "./data"
    
    # 执行清洗
    main(RAW_TEXT_PATH, OUTPUT_DIR)