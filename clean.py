"""
通用版：可切换保留/移除标点，同时生成带标点/无标点两个版本
- 保留标点：用于端到端OCR+标点模型训练
- 移除标点：纯OCR模型训练（彻底移除所有中文/英文标点）
"""
import os
import re

def clean_raw_text(
    raw_text_path: str,
    output_dir: str,
    keep_punctuation: bool = True  # 控制是否保留标点
) -> list:
    """
    清洗逻辑：
    1. 保留多行结构（每行对应图片的一列/一行）
    2. 移除行首数字编号（2、3、4...）
    3. 可选：保留/彻底移除所有标点（中文+英文）
    4. 跳过空行，保留繁体汉字
    
    关键修正：
    - 正则表达式正确转义特殊字符，覆盖所有标点类型
    - 彻底移除无标点版本的所有标点，无遗漏
    """
    # ========== 修正核心：全覆盖+正确转义的标点正则 ==========
    # 包含：所有中文全角标点 + 英文半角标点 + 特殊符号
    punctuation_pattern = re.compile(
        r'[，。！？；：""''""''（）()【】\[\]、—…·《》<>「」『』｛｝￥＄％％＆＆＊＊＋＋－－／／＼＼｜｜～～｀｀､､·。]'
        r'|[,!?:;"\'(){}<>%&*+-/\\|~`.$]'  # 英文标点兜底
    )
    
    # 读取原始文本
    if not os.path.exists(raw_text_path):
        raise FileNotFoundError(f"原始文本文件不存在：{raw_text_path}")
    
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        # 步骤1：去除每行首尾的空格/换行符（彻底清理空白）
        line_stripped = line.strip()
        
        # 步骤2：跳过空行（纯空白行直接过滤）
        if not line_stripped:
            continue
        
        # 步骤3：移除行首数字编号（仅删行首的数字，不影响文本内数字）
        if line_stripped[0].isdigit():
            line_stripped = line_stripped[1:].strip()
            if not line_stripped:
                continue
        
        # 步骤4：核心修正 - 保留/彻底移除标点
        if not keep_punctuation:
            # 彻底替换所有标点为空字符串
            line_stripped = punctuation_pattern.sub('', line_stripped)
            # 移除后再次清理首尾空白（避免标点移除后留空）
            line_stripped = line_stripped.strip()
        
        # 步骤5：再次跳过移除标点后为空的行
        if not line_stripped:
            continue
        
        # 步骤6：保留该行（严格匹配图片的多行结构）
        cleaned_lines.append(line_stripped)
    
    # 生成输出文件名（区分带/无标点）
    suffix = "with_punct" if keep_punctuation else "no_punct"
    output_path = os.path.join(output_dir, f"cleaned_text_{suffix}.txt")
    
    # 保存清洗后的多行文本（每行单独存储，匹配图片行）
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
    
    # 输出验证信息（直观展示清洗效果）
    print(f"✅ 文本清洗完成（{'保留标点' if keep_punctuation else '移除标点'}）！")
    print(f"   - 原始行数：{len(lines)} → 清洗后有效行数：{len(cleaned_lines)}")
    print(f"   - 保存路径：{output_path}")
    # 预览前2行结果，方便你核对
    if cleaned_lines:
        print(f"   - 预览前2行：")
        for i, line in enumerate(cleaned_lines[:2]):
            print(f"     第{i+1}行：{line}")
    
    return cleaned_lines

if __name__ == "__main__":
    # 路径配置（和你的项目结构严格对齐）
    RAW_TEXT_PATH = "./data/raw_text_original.txt"
    OUTPUT_DIR = "./data"
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成两个版本：1. 保留标点（端到端模型） 2. 移除标点（纯OCR模型）
    print("===== 生成带标点版本（端到端OCR+标点） =====")
    clean_raw_text(RAW_TEXT_PATH, OUTPUT_DIR, keep_punctuation=True)
    
    print("\n===== 生成无标点版本（纯OCR） =====")
    clean_raw_text(RAW_TEXT_PATH, OUTPUT_DIR, keep_punctuation=False)