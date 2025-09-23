import os
import argparse
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# 配置
MODEL_DIR = "bart_correction_model"
MODEL_PATH = os.path.join(MODEL_DIR, "final")


def load_model():
    """加载训练好的模型和tokenizer"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

    # 移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def correct_text(model, tokenizer, device, text, max_length=128, num_beams=4):
    """纠正文本"""
    # 编码输入文本
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # 生成纠正后的文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    # 解码输出
    corrected_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return corrected_text


def batch_correct(model, tokenizer, device, texts, max_length=128, num_beams=4):
    """批量纠正文本"""
    # 编码输入文本
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # 生成纠正后的文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    # 解码输出
    corrected_texts = [
        tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for output in outputs
    ]

    return corrected_texts


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用中文纠错模型进行预测")
    parser.add_argument("--text", type=str, help="要纠正的文本")
    parser.add_argument("--file", type=str, help="包含要纠正的文本的文件路径")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--num_beams", type=int, default=4, help="beam search数量")
    args = parser.parse_args()

    # 检查输入
    if not args.text and not args.file:
        parser.error("必须提供--text或--file参数")

    try:
        # 加载模型
        print("加载模型...")
        model, tokenizer, device = load_model()
        print(f"使用设备: {device}")

        # 处理单个文本
        if args.text:
            corrected = correct_text(model, tokenizer, device, args.text, args.max_length, args.num_beams)
            print(f"原始文本: {args.text}")
            print(f"纠正后: {corrected}")

        # 处理文件
        if args.file:
            if not os.path.exists(args.file):
                raise FileNotFoundError(f"文件不存在: {args.file}")

            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            print(f"从文件加载了 {len(texts)} 条文本")
            corrected_texts = batch_correct(model, tokenizer, device, texts, args.max_length, args.num_beams)

            # 保存结果
            output_file = os.path.splitext(args.file)[0] + "_corrected.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for original, corrected in zip(texts, corrected_texts):
                    f.write(f"原始: {original}\n")
                    f.write(f"纠正: {corrected}\n\n")

            print(f"结果已保存到 {output_file}")

    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
