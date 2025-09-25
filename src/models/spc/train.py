import os
import argparse
import torch
import json

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,  # 适配Seq2Seq生成任务的AutoModel头
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm import tqdm  # 进度条（需安装：pip install tqdm）
import warnings
warnings.filterwarnings("ignore")  # 忽略无关警告

# -------------------------- 基础配置（与process.py对齐） --------------------------
# 处理后数据目录（process.py的输出目录）
PROCESSED_DATA_DIR = "D:\\python_project\\test\\ai_medical\\data\\spell_check\\processed"
# 模型保存目录（训练完成后保存模型）
MODEL_SAVE_DIR = "D:\\python_project\\test\\ai_medical\\models\\bart_correction"
# 预训练模型（中文BART基础模型，适配中文纠错任务）
PRETRAINED_MODEL = "D:\\python_project\\test\\ai_medical\\pretrained\\bart-base-chinese"
# 设备配置（优先GPU，无GPU则用CPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 文本最大长度（根据你的句子长度调整，BART-base建议≤128）
MAX_LENGTH = 128
# 随机种子（保证训练可复现）
SEED = 42
set_seed(SEED)  # 固定所有随机种子


# -------------------------- 1. 数据集类（适配process.py输出的JSON格式） --------------------------
class CorrectionDataset(Dataset):
    """
    中文文本纠错数据集：读取process.py生成的train.json/val.json/test.json
    每条样本格式：{"input": 错误句, "target": 正确句, "source_file": "...", "line_number": ...}
    """
    def __init__(self, data_path: str, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer  # BART的Tokenizer（用于文本编码）
        # 读取JSON数据（调用process.py生成的文件）
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # 过滤无效样本（双重保险，避免process.py漏筛的空样本）
        self.data = [
            sample for sample in self.data
            if isinstance(sample.get("input"), str) and isinstance(sample.get("target"), str)
            and sample["input"].strip() and sample["target"].strip()
        ]
        print(f"✅ 加载数据集：{os.path.basename(data_path)} | 有效样本数：{len(self.data)}")

    def __len__(self) -> int:
        """返回数据集总样本数"""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        按索引获取单条样本，返回编码后的tensor
        :param idx: 样本索引
        :return: 包含input_ids、attention_mask、labels的字典
        """
        sample = self.data[idx]
        input_text = sample["input"]    # 模型输入：错误句
        target_text = sample["target"]  # 模型输出：正确句

        # 1. 编码输入文本（错误句）：添加BART的特殊token（<s>开头，</s>结尾）
        input_encoding = self.tokenizer(
            input_text,
            max_length=MAX_LENGTH,
            padding="max_length",  # 不足MAX_LENGTH补0
            truncation=True,       # 超过MAX_LENGTH截断
            return_tensors="pt"    # 返回PyTorch tensor
        )

        # 2. 编码目标文本（正确句）：BART的labels无需attention_mask，仅需input_ids
        target_encoding = self.tokenizer(
            target_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 挤压维度（去除batch维度，DataLoader会自动添加batch）
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),  # (MAX_LENGTH,)
            "attention_mask": input_encoding["attention_mask"].squeeze(0),  # (MAX_LENGTH,)
            "labels": target_encoding["input_ids"].squeeze(0)  # (MAX_LENGTH,)，用于计算损失
        }


# -------------------------- 2. 数据加载函数（创建DataLoader，支持批量训练） --------------------------
def create_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """
    创建DataLoader（批量加载数据，支持多线程加速）
    :param data_path: JSON数据文件路径（如train.json）
    :param tokenizer: AutoTokenizer实例
    :param batch_size: 批次大小（根据GPU显存调整，16GB显存建议8-16）
    :param shuffle: 是否打乱数据（训练集True，验证/测试集False）
    :return: DataLoader实例
    """
    dataset = CorrectionDataset(data_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,  # 多线程读取（CPU核心数≥2时建议设2，避免卡顿）
        pin_memory=True,  # 加速GPU数据传输（有GPU时启用）
        drop_last=False  # 不丢弃最后一个不足batch的样本
    )
    return dataloader


# -------------------------- 3. 训练核心函数（单轮epoch训练） --------------------------
def train_one_epoch(
    model: AutoModelForSeq2SeqLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    args
) -> float:
    """
    训练一个epoch（遍历一次训练集）
    :param model: 训练的模型
    :param dataloader: 训练集DataLoader
    :param optimizer: 优化器（如AdamW）
    :param scheduler: 学习率调度器（控制学习率衰减）
    :param epoch: 当前epoch编号（从0开始）
    :param args: 命令行参数（含epochs、batch_size等）
    :return: 该epoch的平均训练损失
    """
    model.train()  # 切换模型为训练模式（启用Dropout等）
    total_loss = 0.0  # 累计总损失
    # 进度条（显示当前epoch、批次、损失）
    progress_bar = tqdm(
        dataloader,
        desc=f"📌 Train Epoch {epoch+1}/{args.epochs}",
        unit="batch",
        colour="green"
    )

    for batch in progress_bar:
        # 1. 将batch数据移动到GPU/CPU
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        # 2. 清零梯度（避免梯度累积）
        optimizer.zero_grad()

        # 3. 前向传播：模型输出logits和损失（AutoModelForSeq2SeqLM自动计算交叉熵损失）
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss  # 模型内置的损失（基于labels计算）

        # 4. 反向传播：计算梯度
        loss.backward()

        # 5. 梯度裁剪（防止梯度爆炸，BART模型常用）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. 优化器更新参数
        optimizer.step()

        # 7. 学习率调度器更新学习率
        scheduler.step()

        # 8. 累计损失并更新进度条
        batch_loss = loss.item()
        total_loss += batch_loss * input_ids.size(0)  # 按样本数加权累计
        progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})  # 实时显示批次损失

    # 计算该epoch的平均损失（总损失/总样本数）
    avg_train_loss = total_loss / len(dataloader.dataset)
    print(f"✅ Train Epoch {epoch+1} | Avg Loss: {avg_train_loss:.4f}\n")
    return avg_train_loss


# -------------------------- 4. 验证核心函数（单轮epoch验证，无梯度更新） --------------------------
def val_one_epoch(
    model: AutoModelForSeq2SeqLM,
    dataloader: DataLoader,
    epoch: int,
    args
) -> float:
    """
    验证一个epoch（遍历一次验证集，不更新模型参数）
    :param model: 训练的模型
    :param dataloader: 验证集DataLoader
    :param epoch: 当前epoch编号（从0开始）
    :param args: 命令行参数
    :return: 该epoch的平均验证损失
    """
    model.eval()  # 切换模型为评估模式（禁用Dropout等）
    total_loss = 0.0
    progress_bar = tqdm(
        dataloader,
        desc=f"📌 Val Epoch {epoch+1}/{args.epochs}",
        unit="batch",
        colour="blue"
    )

    # 禁用梯度计算（加速验证，节省内存）
    with torch.no_grad():
        for batch in progress_bar:
            # 数据移动到设备
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            # 前向传播（仅计算损失，不反向传播）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            batch_loss = loss.item()
            total_loss += batch_loss * input_ids.size(0)

            # 更新进度条
            progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

    # 计算平均验证损失
    avg_val_loss = total_loss / len(dataloader.dataset)
    print(f"✅ Val Epoch {epoch+1} | Avg Loss: {avg_val_loss:.4f}\n")
    return avg_val_loss


# -------------------------- 5. 模型保存函数（保存最优模型和Tokenizer） --------------------------
def save_best_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    save_dir: str,
    epoch: int,
    val_loss: float
):
    """
    保存验证损失最优的模型（避免保存所有epoch的模型，节省空间）
    :param model: 训练的模型
    :param tokenizer: AutoTokenizer实例（必须与模型配套保存）
    :param save_dir: 模型保存目录
    :param epoch: 当前epoch编号
    :param val_loss: 当前epoch的验证损失
    """
    # 创建保存目录（若不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 保存路径（标注最优模型的epoch和损失）
    best_model_dir = os.path.join(save_dir, f"best_model_epoch{epoch+1}_valloss{val_loss:.4f}")
    # 保存模型和Tokenizer（Hugging Face标准格式，后续predict.py可直接加载）
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"🏆 保存最优模型到：{best_model_dir}\n")


# -------------------------- 6. 主函数（整合全流程：加载→训练→验证→保存） --------------------------
def main():
    # 解析命令行参数（支持动态调整训练参数，无需修改代码）
    parser = argparse.ArgumentParser(description="训练中文文本纠错BART模型（适配process.py输出）")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数（建议10-20，根据数据量调整）")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小（16GB显存建议8，24GB建议16）")
    parser.add_argument("--lr", type=float, default=3e-5, help="初始学习率（BART常用3e-5~5e-5）")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="学习率预热比例（前10%步数缓慢升LR）")
    args = parser.parse_args()

    # 打印训练配置信息
    print("=" * 70)
    print("📋 训练配置信息")
    print("=" * 70)
    print(f"预训练模型：{PRETRAINED_MODEL}")
    print(f"设备：{DEVICE}（GPU型号：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}）")
    print(f"训练轮数：{args.epochs} | 批次大小：{args.batch_size} | 初始学习率：{args.lr}")
    print(f"训练集路径：{os.path.join(PROCESSED_DATA_DIR, 'train.json')}")
    print(f"验证集路径：{os.path.join(PROCESSED_DATA_DIR, 'val.json')}")
    print(f"模型保存路径：{MODEL_SAVE_DIR}")
    print("=" * 70)

    try:
        # -------------------------- 步骤1：加载Tokenizer和预训练模型 --------------------------
        print("\n📥 加载Tokenizer和预训练模型...")
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        # 加载Seq2Seq生成模型（AutoModelForSeq2SeqLM自动适配BART的生成头）
        model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
        model = model.to(DEVICE)  # 移动模型到设备
        print("✅ 模型和Tokenizer加载完成！")

        # -------------------------- 步骤2：创建训练集/验证集DataLoader --------------------------
        print("\n📥 创建DataLoader...")
        train_loader = create_dataloader(
            data_path=os.path.join(PROCESSED_DATA_DIR, "train.json"),
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            shuffle=True  # 训练集打乱
        )
        val_loader = create_dataloader(
            data_path=os.path.join(PROCESSED_DATA_DIR, "val.json"),
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            shuffle=False  # 验证集不打乱
        )

        # -------------------------- 步骤3：初始化优化器和学习率调度器 --------------------------
        print("\n📥 初始化优化器和学习率调度器...")
        # 优化器：AdamW（BART模型常用，支持权重衰减）
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01  # 权重衰减（防止过拟合）
        )
        # 总训练步数 =  epoch数 × 每epoch批次数量
        total_training_steps = args.epochs * len(train_loader)
        # 预热步数 = 总步数 × 预热比例（前10%步数学习率从0升到初始LR）
        warmup_steps = int(total_training_steps * args.warmup_ratio)
        # 学习率调度器：线性预热+线性衰减（BART训练常用策略）
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        print(f"✅ 优化器初始化完成！总训练步数：{total_training_steps} | 预热步数：{warmup_steps}")

        # -------------------------- 步骤4：开始训练+验证循环 --------------------------
        print("\n🚀 开始训练...")
        best_val_loss = float("inf")  # 初始化最优验证损失（无穷大）
        for epoch in range(args.epochs):
            # 训练一个epoch
            train_one_epoch(model, train_loader, optimizer, scheduler, epoch, args)
            # 验证一个epoch
            val_loss = val_one_epoch(model, val_loader, epoch, args)
            # 保存最优模型（仅当当前验证损失低于历史最优时保存）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_model(model, tokenizer, MODEL_SAVE_DIR, epoch, best_val_loss)

        # -------------------------- 步骤5：训练完成总结 --------------------------
        print("=" * 70)
        print("🎉 训练全部完成！")
    finally:
        print("\n📝 训练流程结束，清理资源...")
        # 清理GPU内存（避免显存占用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ 资源清理完成！")
if __name__ == '__main__':
    main()