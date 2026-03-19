import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ========== 核心配置 ==========
MODEL_PATH = "/root/wangdingfu-handwritten-ocr/models/qwen-7b-chat/qwen/Qwen-7B-Chat"
TRAIN_FILE = "/root/wangdingfu-handwritten-ocr/data/train_no_punct.jsonl"
VAL_FILE = "/root/wangdingfu-handwritten-ocr/data/val_no_punct.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/qwen_7b_ocr_no_punct"
MAX_LEN = 512
PROMPT = "识别手写古籍文献："

# --- 原生 CER 计算函数 ---
def calculate_cer(preds, labels):
    distances = 0
    total_chars = 0
    for p, l in zip(preds, labels):
        # 简单的编辑距离实现
        if len(p) < len(l): p, l = l, p
        if len(l) == 0: distances += len(p); total_chars += len(p); continue
        prev = range(len(l) + 1)
        for i, c1 in enumerate(p):
            curr = [i + 1]
            for j, c2 in enumerate(l):
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (c1 != c2)))
            prev = curr
        distances += prev[-1]
        total_chars += len(l)
    return distances / max(total_chars, 1)

# --- 数据集类 ---
class OCRDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data[idx]
        full_text = f"{PROMPT}{target}{self.tokenizer.eos_token}"
        
        model_inputs = self.tokenizer(
            full_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = model_inputs["input_ids"].squeeze()
        labels = input_ids.clone()
        
        # 找到 Prompt 的长度，将 Prompt 对应的 Labels 设为 -100 (不计算 Loss)
        prompt_ids = self.tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
        labels[:len(prompt_ids)] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": model_inputs["attention_mask"].squeeze()
        }

# --- 评估指标 ---
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
    # 替换掉填充值
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 清理 Prompt 字符
    clean_preds = [p.replace(PROMPT, "").strip() for p in decoded_preds]
    clean_labels = [l.replace(PROMPT, "").strip() for l in decoded_labels]

    cer = calculate_cer(clean_preds, clean_labels)
    return {"accuracy": 1 - cer, "cer": cer}

def main():
    global tokenizer
    # 1. 强制注入 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False,
        pad_token='<|endoftext|>', eos_token='<|endoftext|>'
    )
    tokenizer.pad_token_id = 151643
    tokenizer.padding_side = "left"

    # 2. 加载数据并手动切分 Test
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_list = [json.loads(line)["text"] for line in f]
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        full_val_list = [json.loads(line)["text"] for line in f]
    
    # 选取最后一条作为 Test，其余作为 Val
    test_text = full_val_list[-1]
    val_list = full_val_list[:-1]

    train_dataset = OCRDataset(train_list, tokenizer)
    val_dataset = OCRDataset(val_list, tokenizer)

    # 3. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # 4. 训练配置
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=5, # 演示用 5 轮，可自行增加
        learning_rate=1e-5,
        bf16=True,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=20,
        predict_with_generate=True,
        generation_max_length=MAX_LEN,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
        compute_metrics=compute_metrics
    )

    print(f"✅ 数据状态: 训练集 {len(train_list)}, 验证集 {len(val_list)}, 测试 1 条")
    print("🚀 开始训练...")
    trainer.train()

    # 5. 针对那条 Test 数据的专项测试
    print("\n" + "="*30)
    print("🎯 进行独立测试 (Test Case)")
    model.eval()
    inputs = tokenizer(f"{PROMPT}", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_LEN, pad_token_id=tokenizer.pad_token_id)
    
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(PROMPT, "").strip()
    
    print(f"【标准答案】: {test_text}")
    print(f"【模型预测】: {pred_text}")
    
    test_cer = calculate_cer([pred_text], [test_text])
    print(f"【单条准确率】: {1 - test_cer:.4f}")
    print("="*30)

    # 保存
    trainer.save_model(OUTPUT_DIR)
    print(f"🎉 任务完成，模型保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()