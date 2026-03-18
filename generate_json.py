"""
通用版：生成带标点/无标点两个版本的JSONL数据集
- 带标点：train_with_punct.jsonl / val_with_punct.jsonl（端到端模型）
- 无标点：train_no_punct.jsonl / val_no_punct.jsonl（纯OCR模型）
关键修正：
- 强化路径校验，确保读取的是正确的清洗文件
- 增加文本行校验，避免空行/无效行混入
"""
import json
import os
from typing import List, Dict

def build_ocr_dataset(
    image_dir: str,
    cleaned_text_path: str,
    output_dir: str,
    val_ratio: float = 0.1
) -> None:
    """
    生成数据集，每张图片对应多行文本（匹配竖排图片结构）
    
    Args:
        image_dir: 本地图片文件夹路径
        cleaned_text_path: 清洗后的文本路径（带/无标点）
        output_dir: 数据集输出目录
        val_ratio: 验证集比例
    """
    # 1. 严格校验输入文件（避免读取错误文件）
    if not os.path.exists(cleaned_text_path):
        raise FileNotFoundError(
            f"清洗后的文本文件不存在：{cleaned_text_path}\n"
            f"请先运行 text_cleaner.py 生成对应文件！"
        )
    
    # 2. 读取清洗后的多行文本（过滤空行，确保纯有效文本）
    with open(cleaned_text_path, 'r', encoding='utf-8') as f:
        cleaned_lines = []
        for line in f.readlines():
            line_clean = line.strip()
            if line_clean:  # 彻底过滤空行
                cleaned_lines.append(line_clean)
    
    if not cleaned_lines:
        raise ValueError(f"清洗后的文本文件为空：{cleaned_text_path}")
    
    # 3. 获取排序后的图片文件（按文件名升序，确保匹配顺序）
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    if not image_files:
        raise ValueError(f"图片文件夹为空：{image_dir}，请放入jpg格式的手写图片！")
    
    image_files.sort()  # 严格按文件名排序（00002.jpg → 00012.jpg）
    total_images = len(image_files)
    
    # 4. 配置每张图片对应的文本行数（请根据你的实际图片调整！）
    lines_per_image = 8  # 示例：每张图片对应8行文本，需替换为你的实际行数
    total_text_lines = len(cleaned_lines)
    total_matched_images = total_text_lines // lines_per_image
    
    # 校验文本行数是否足够
    if total_matched_images == 0:
        raise ValueError(
            f"文本行数不足！\n"
            f"每张图片需{lines_per_image}行 → 共需{total_images * lines_per_image}行\n"
            f"当前仅找到{total_text_lines}行，请补充文本或调整lines_per_image！"
        )
    if total_matched_images < total_images:
        print(f"⚠️  警告：文本行数仅匹配{total_matched_images}张图片（共{total_images}张），剩余图片将被忽略！")
    
    # 5. 图片 ↔ 多行文本 精准匹配（按行分配，无错位）
    dataset = []
    for i, img_file in enumerate(image_files[:total_matched_images]):
        # 截取当前图片对应的文本行（严格按行数分配）
        start_idx = i * lines_per_image
        end_idx = start_idx + lines_per_image
        image_text_lines = cleaned_lines[start_idx:end_idx]
        
        # 拼接为换行分隔的字符串（保留多行结构，适配竖排OCR）
        image_text = '\n'.join(image_text_lines)
        
        # 生成数据集条目（增加版本标记，方便后续管理）
        dataset.append({
            "image": os.path.join("local_images", img_file),  # 相对路径，适配训练环境
            "text": image_text,                              # 多行文本（带/无标点）
            "line_count": len(image_text_lines),              # 标注行数，方便核对
            "punctuation_version": "with_punct" if "with_punct" in cleaned_text_path else "no_punct"
        })
    
    # 6. 拆分训练集/验证集（至少保留1个验证样本）
    val_size = max(1, int(len(dataset) * val_ratio))
    train_set = dataset[val_size:]
    val_set = dataset[:val_size]
    
    # 7. 生成版本化的输出文件名（避免覆盖）
    version_suffix = "with_punct" if "with_punct" in cleaned_text_path else "no_punct"
    train_output = os.path.join(output_dir, f"train_{version_suffix}.jsonl")
    val_output = os.path.join(output_dir, f"val_{version_suffix}.jsonl")
    
    # 8. 保存JSONL文件（确保编码正确，无乱码）
    def save_jsonl(data: List[Dict], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                # ensure_ascii=False 保留中文，避免转义
                json.dump(entry, f, ensure_ascii=False, indent=None)
                f.write('\n')
    
    save_jsonl(train_set, train_output)
    save_jsonl(val_set, val_output)
    
    # 输出清晰的生成结果
    print(f"✅ 数据集生成完成（{version_suffix}版本）！")
    print(f"   - 匹配图片数：{len(dataset)} 张")
    print(f"   - 训练集：{len(train_set)} 样本 → {train_output}")
    print(f"   - 验证集：{len(val_set)} 样本 → {val_output}")
    # 预览第一个样本，方便核对
    if dataset:
        print(f"   - 预览第一个样本：")
        print(f"     图片路径：{dataset[0]['image']}")
        print(f"     文本内容（前50字）：{dataset[0]['text'][:50]}...")

if __name__ == "__main__":
    # 路径配置（和你的项目结构严格对齐）
    IMAGE_DIR = "./local_images"
    OUTPUT_DIR = "./data"
    DATA_DIR = "./data"
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成两个版本的数据集（严格按清洗后的文件匹配）
    print("===== 生成带标点数据集（端到端OCR+标点） =====")
    build_ocr_dataset(
        image_dir=IMAGE_DIR,
        cleaned_text_path=os.path.join(DATA_DIR, "cleaned_text_with_punct.txt"),
        output_dir=OUTPUT_DIR
    )
    
    print("\n===== 生成无标点数据集（纯OCR） =====")
    build_ocr_dataset(
        image_dir=IMAGE_DIR,
        cleaned_text_path=os.path.join(DATA_DIR, "cleaned_text_no_punct.txt"),
        output_dir=OUTPUT_DIR
    )