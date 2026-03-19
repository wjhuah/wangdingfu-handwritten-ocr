#!/bin/bash

# 确保在项目根目录
cd /root/wangdingfu-handwritten-ocr

# 1. 运行训练
echo "🚀 启动训练程序..."
python scripts/train_qwen_ocr_simple.py

# 2. 检查 Log 是否真的存在
LOG_SRC="/root/autodl-tmp/qwen_7b_ocr_no_punct/train_log.txt"
LOG_DEST="./scripts/latest_train_log.txt"

if [ -f "$LOG_SRC" ]; then
    cp "$LOG_SRC" "$LOG_DEST"
    echo "📄 Log 文件已同步至仓库。"
else
    echo "❌ 错误: 未找到训练日志 $LOG_SRC，请检查训练是否成功完成！"
    exit 1
fi

# 3. Git 推送
echo "📦 准备推送至 GitHub..."
git add .
# 获取准确率作为 Commit Message
ACC=$(grep "单条准确率" "$LOG_DEST" | awk '{print $NF}')
git commit -m "Update OCR train log - Accuracy: $ACC"
git push origin main

echo "✨ 所有任务已完成！"