"""
适配2-79图片编号：
- 提取图片文件名中的数字（0002.jpg→2，0079.jpg→79）
- 按编号匹配清洗后的文本块
- 生成带标点/无标点JSONL数据集
"""
import json
import os
import re
from typing import List, Dict

# ========== 核心配置（无需修改） ==========
START_NUM = 2    # 起始图片编号
END_NUM = 79     # 结束图片编号
IMAGE_NUM_PATTERN = re.compile(r'(\d+)')  # 提取图片文件名中的数字

def load_cleaned_text_blocks(cleaned_text_path: str) -> Dict[int, List[str]]:
    """
    加载清洗后的文本块（从cleaned_text_blocks_*.txt读取）
    返回：{图片号: 文本行列表}
    """
    if not os.path.exists(cleaned_text_path):
        raise FileNotFoundError(f"清洗后的文本文件不存在：{cleaned_text_path}\n请先运行text_cleaner.py！")
    
    image_text_map = {}
    current_num = None
    current_lines = []
    
    with open(cleaned_text_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_stripped = line.strip()
            # 识别图片编号行（===== 图片编号：2（0002.jpg）=====）
            if line_stripped.startswith("===== 图片编号："):
                # 保存上一个图片的文本
                if current_num is not None and current_lines:
                    image_text_map[current_num] = current_lines
                # 提取图片编号
                num_match = IMAGE_NUM_PATTERN.search(line_stripped)
                if num_match:
                    current_num = int(num_match.group(1))
                    current_lines = []
                continue
            # 收集文本行
            if current_num is not None and line_stripped:
                current_lines.append(line_stripped)
    # 保存最后一个图片的文本
    if current_num is not None and current_lines:
        image_text_map[current_num] = current_lines
    
    return image_text_map

def build_ocr_dataset(
    image_dir: str,
    cleaned_text_path: str,
    output_dir: str,
    val_ratio: float = 0.1
) -> None:
    """
    生成JSONL数据集：按图片编号匹配文本块
    """
    # 步骤1：加载清洗后的文本块
    image_text_map = load_cleaned_text_blocks(cleaned_text_path)
    # 步骤2：获取图片文件并提取编号
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    if not image_files:
        raise ValueError(f"图片文件夹为空：{image_dir}，请放入2-79的jpg图片！")
    
    # 按图片编号排序（0002.jpg→2，0079.jpg→79）
    image_file_map = {}  # {图片号: 文件名}
    for img_file in image_files:
        num_match = IMAGE_NUM_PATTERN.search(img_file)
        if num_match:
            num = int(num_match.group(1))
            if START_NUM <= num <= END_NUM:
                image_file_map[num] = img_file
    # 按编号排序图片
    sorted_nums = sorted(image_file_map.keys())
    sorted_image_files = [image_file_map[num] for num in sorted_nums]
    
    # 步骤3：匹配图片和文本
    dataset = []
    for num in sorted_nums:
        if num not in image_text_map:
            print(f"⚠️  警告：图片{num}（{image_file_map[num]}）无匹配文本，跳过！")
            continue
        # 获取文本块并拼接为多行字符串
        text_lines = image_text_map[num]
        text = '\n'.join(text_lines)
        # 生成数据集条目
        dataset.append({
            "image": os.path.join("local_images", image_file_map[num]),
            "text": text,
            "image_number": num,          # 图片编号（2-79）
            "text_line_count": len(text_lines),  # 文本行数
            "punctuation_version": "with_punct" if "with_punct" in cleaned_text_path else "no_punct"
        })
    
    if not dataset:
        raise ValueError("无匹配的图片-文本对！请检查图片编号和文本编号是否一致。")
    
    # 步骤4：拆分训练集/验证集（至少1个验证样本）
    val_size = max(1, int(len(dataset) * val_ratio))
    train_set = dataset[val_size:]
    val_set = dataset[:val_size]
    
    # 步骤5：保存JSONL文件
    version_suffix = "with_punct" if "with_punct" in cleaned_text_path else "no_punct"
    train_output = os.path.join(output_dir, f"train_{version_suffix}.jsonl")
    val_output = os.path.join(output_dir, f"val_{version_suffix}.jsonl")
    
    def save_jsonl(data: List[Dict], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
    
    save_jsonl(train_set, train_output)
    save_jsonl(val_set, val_output)
    
    # 输出结果
    print(f"✅ 数据集生成完成（{version_suffix}版本）！")
    print(f"   - 匹配样本数：{len(dataset)}（图片{START_NUM}-{END_NUM}）")
    print(f"   - 训练集：{len(train_set)} 样本 → {train_output}")
    print(f"   - 验证集：{len(val_set)} 样本 → {val_output}")
    # 预览第一个样本
    print(f"   - 预览第一个样本：")
    print(f"     图片：{dataset[0]['image']}（编号{dataset[0]['image_number']}）")
    print(f"     文本（前50字）：{dataset[0]['text'][:50]}...")

if __name__ == "__main__":
    # 路径配置（和你的项目结构对齐）
    IMAGE_DIR = "./local_images"
    OUTPUT_DIR = "./data"
    DATA_DIR = "./data"
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成带标点数据集（端到端模型）
    print("===== 生成带标点数据集（端到端OCR+标点） =====")
    build_ocr_dataset(
        image_dir=IMAGE_DIR,
        cleaned_text_path=os.path.join(DATA_DIR, "cleaned_text_blocks_with_punct.txt"),
        output_dir=OUTPUT_DIR
    )
    
    # 生成无标点数据集（纯OCR模型）
    print("\n===== 生成无标点数据集（纯OCR） =====")
    build_ocr_dataset(
        image_dir=IMAGE_DIR,
        cleaned_text_path=os.path.join(DATA_DIR, "cleaned_text_blocks_no_punct.txt"),
        output_dir=OUTPUT_DIR
    )