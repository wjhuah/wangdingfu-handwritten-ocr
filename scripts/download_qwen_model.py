"""
从ModelScope（国内源）下载Qwen 3.5 7B Chat模型
修正正则报错，简化下载逻辑，适配AutoDL服务器
"""
from modelscope.hub.snapshot_download import snapshot_download
import os

# ========== 配置项 ==========
MODEL_NAME = "qwen/Qwen-7B-Chat"  # ModelScope上的Qwen 3.5 7B地址
SAVE_PATH = "./models/qwen-7b-chat"  # 模型保存路径

# 创建模型保存目录
os.makedirs(SAVE_PATH, exist_ok=True)

# 下载模型（自动用国内源，支持断点续传）
if __name__ == "__main__":
    print(f"===== 开始下载Qwen 3.5 7B模型（国内源）=====")
    print(f"模型保存路径：{SAVE_PATH}")
    
    # 核心：从ModelScope下载（去掉有问题的ignore_file_pattern）
    snapshot_download(
        MODEL_NAME,
        cache_dir=SAVE_PATH,
        revision="master"  # 移除ignore_file_pattern，避免正则报错
    )
    
    print(f"✅ Qwen 3.5 7B模型下载完成！路径：{SAVE_PATH}")