import os
import argparse
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
import evaluate
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# -------------------------- 配置参数 --------------------------
PROCESSED_DATA_DIR = "D:\\python_project\\test\\ai_medical\\data\\spell_check\\processed"
MODEL_SAVE_DIR = "D:\\python_project\\test\\ai_medical\\models\\bart_correction"
PRETRAINED_MODEL = "fnlp/bart-base-chinese"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
SEED = 42
set_seed(SEED)


# -------------------------- 数据集类 --------------------------
class CorrectionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 过滤无效样本
        self.data = [
            sample for sample in self.data
            if isinstance(sample.get("input"), str) and isinstance(sample.get("target"), str)
               and sample["input"].strip() and sample["target"].strip()
        ]
        print(f"加载数据集：{os.path.basename(data_path)} | 有效样本数：{len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        input_text = sample["input"]
        target_text = sample["target"]

        # 编码输入和目标文本
        inputs = self.tokenizer(
            input_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer(
            target_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 转换为字典并挤压维度
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0)
        }

        # 处理标签中的pad token（设置为-100以忽略计算损失）
        item["labels"][item["labels"] == self.tokenizer.pad_token_id] = -100
        return item


# -------------------------- 评估指标函数 --------------------------
def compute_metrics(eval_preds):
    """计算BLEU和ROUGE指标"""
    metric_bleu = evaluate.load("bleu")
    metric_rouge = evaluate.load("rouge")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 解码预测和标签
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 计算BLEU分数
    bleu_results = metric_bleu.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels],  # BLEU需要列表的列表
        max_order=4
    )

    # 计算ROUGE分数
    rouge_results = metric_rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {
        "bleu": round(bleu_results["bleu"], 4),
        "rouge-l": round(rouge_results["rougeL"], 4)
    }


# -------------------------- 主函数 --------------------------
def main():
    parser = argparse.ArgumentParser(description="使用Trainer训练中文文本纠错模型")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-5, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=500, help="预热步数")
    args = parser.parse_args()

    # 打印配置信息
    print("=" * 70)
    print("训练配置信息")
    print("=" * 70)
    print(f"模型: {PRETRAINED_MODEL}")
    print(f"设备: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"训练参数: 轮数={args.epochs}, 批次大小={args.batch_size}, 学习率={args.lr}")
    print(f"数据路径: {PROCESSED_DATA_DIR}")
    print(f"模型保存路径: {MODEL_SAVE_DIR}")
    print("=" * 70)

    try:
        # 加载tokenizer和模型
        print("\n加载Tokenizer和预训练模型...")
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
        model = model.to(DEVICE)

        # 加载数据集
        print("\n加载数据集...")
        train_dataset = CorrectionDataset(
            os.path.join(PROCESSED_DATA_DIR, "train.json"),
            tokenizer
        )
        val_dataset = CorrectionDataset(
            os.path.join(PROCESSED_DATA_DIR, "val.json"),
            tokenizer
        )

        # 定义数据收集器（处理批次数据）
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        # 定义训练参数
        training_args = TrainingArguments(
            output_dir=MODEL_SAVE_DIR,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            weight_decay=0.01,  # 权重衰减
            logging_dir=f"{MODEL_SAVE_DIR}/logs",  # 日志目录
            logging_steps=10,  # 每10步打印一次日志
            evaluation_strategy="epoch",  # 每个epoch评估一次
            save_strategy="epoch",  # 每个epoch保存一次模型
            save_total_limit=3,  # 最多保存3个模型
            load_best_model_at_end=True,  # 训练结束后加载最佳模型
            metric_for_best_model="rouge-l",  # 以rouge-l作为最佳模型指标
            fp16=torch.cuda.is_available(),  # 若有GPU则启用混合精度训练
            remove_unused_columns=False,  # 保留自定义数据集的所有字段
        )

        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,  # 评估指标计算函数
        )

        # 开始训练
        print("\n开始训练...")
        trainer.train()

        # 在测试集上评估
        print("\n在测试集上评估最佳模型...")
        test_dataset = CorrectionDataset(
            os.path.join(PROCESSED_DATA_DIR, "test.json"),
            tokenizer
        )
        test_results = trainer.evaluate(eval_dataset=test_dataset)

        print("\n测试集评估结果:")
        for key, value in test_results.items():
            if "bleu" in key or "rouge" in key:
                print(f"{key}: {value:.4f}")

        print("\n训练完成！最佳模型已保存至:", MODEL_SAVE_DIR)

    except Exception as e:
        print(f"\n训练过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n资源清理完成")


if __name__ == "__main__":
    main()
