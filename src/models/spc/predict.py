import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")


def text_correction_predict(
        text: str | list[str],
        model_dir: str,
        max_length: int = 128,
        batch_size: int = 8,
        num_beams: int = 5
) -> str | list[str]:
    """
    独立的中文文本纠错预测方法：支持单条/批量文本输入，返回纠错后的结果

    参数说明：
    ----------
    text: str | list[str]
        待纠错的文本，支持两种输入格式：
        - 单条文本：如 "我今天感帽了，需要喝药"
        - 批量文本：如 ["感帽", "咳簌", "发绕，头通"]
    model_dir: str
        训练好的模型保存目录（需包含pytorch_model.bin、config.json、tokenizer.json等文件）
    max_length: int, default=128
        文本最大编码长度（需与训练时保持一致，避免格式不兼容）
    batch_size: int, default=8
        批量预测时的批次大小（根据GPU显存调整：16GB显存建议8-16，CPU建议2-4）
    num_beams: int, default=5
        Beam Search数量（越大纠错效果越优，但速度越慢；推荐范围3-10）

    返回值：
    ----------
    str | list[str]
        纠错后的文本，与输入格式对应：
        - 输入单条文本 → 返回单条纠错结果（str）
        - 输入批量文本 → 返回批量纠错结果（list[str]）

    示例：
    ----------
    # 1. 单条文本纠错
    result = text_correction_predict(
        text="我今天感帽了",
        model_dir="D:\\models\\bart_correction\\best_model"
    )
    print(result)  # 输出："我今天感冒了"

    # 2. 批量文本纠错
    results = text_correction_predict(
        text=["感帽", "咳簌", "发绕"],
        model_dir="D:\\models\\bart_correction\\best_model",
        batch_size=3
    )
    print(results)  # 输出：["感冒", "咳嗽", "发烧"]
    """
    # -------------------------- 1. 初始化设备、模型、Tokenizer --------------------------
    # 自动选择设备（优先GPU，无GPU则用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和Tokenizer（与训练时的Auto类完全一致，避免类型不匹配）
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()  # 切换为评估模式（禁用Dropout，保证预测稳定）

    # -------------------------- 2. 统一输入格式（单条→列表，便于批量处理） --------------------------
    is_single_input = isinstance(text, str)
    if is_single_input:
        text_list = [text.strip()]  # 单条文本转为列表
        if not text_list[0]:
            raise ValueError("待纠错文本不能为空！")
    else:
        # 批量输入：过滤空字符串，统一格式
        text_list = [t.strip() for t in text if isinstance(t, str) and t.strip()]
        if not text_list:
            raise ValueError("批量待纠错文本不能为空，且需为字符串列表！")

    # -------------------------- 3. 生成参数配置（平衡纠错效果与速度） --------------------------
    generation_config = {
        "max_length": max_length,
        "min_length": 1,
        "num_beams": num_beams,
        "early_stopping": True,  # 生成EOS或达到max_length时停止
        "no_repeat_ngram_size": 2,  # 避免重复2-gram短语（减少病句）
        "temperature": 1.0,  # 温度：0-1，越小越确定，越大越随机
        "top_p": 1.0,  # Nucleus采样：配合temperature使用，默认1.0
    }

    # -------------------------- 4. 批量预测（避免显存溢出，支持大列表输入） --------------------------
    corrected_results = []
    # 按批次切割文本列表
    for i in range(0, len(text_list), batch_size):
        batch_text = text_list[i:i + batch_size]

        # 文本编码（与训练时编码逻辑完全一致）
        inputs = tokenizer(
            batch_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 数据移动到设备
        input_ids = inputs["input_ids"].to(device, non_blocking=True)
        attention_mask = inputs["attention_mask"].to(device, non_blocking=True)

        # 模型预测（禁用梯度计算，加速+省内存）
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )

        # 解码结果（跳过BART的特殊符号：<s>、</s>、<pad>）
        batch_corrected = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True  # 清理多余空格
        )

        corrected_results.extend(batch_corrected)

    # -------------------------- 5. 清理资源+返回结果（与输入格式对齐） --------------------------
    # 清理GPU内存（避免长期占用）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 单条输入返回字符串，批量输入返回列表
    return corrected_results[0] if is_single_input else corrected_results


# -------------------------- 方法调用示例 --------------------------
if __name__ == "__main__":
    # 配置模型路径（替换为你训练保存的"best_model_xxx"目录）
    # BEST_MODEL_DIR = "D:\\python_project\\test\\ai_medical\\models\\bart_correction\\best_model_epoch1_valloss0.7263"
    BEST_MODEL_DIR="D:\\python_project\\test\\ai_medical\\pretrained\\bart-base-chinese"
    # 1. 单条文本纠错
    single_text = "城府宫员表示,这是过去三十六小时内第三期强烈的余震。"
    single_result = text_correction_predict(
        text=single_text,
        model_dir=BEST_MODEL_DIR,
        num_beams=5
    )
    print("=" * 60)
    print("【单条文本纠错结果】")
    print(f"原始错误文本：{single_text}")
    print(f"纠错后文本：{single_result}")
    print("=" * 60)

    # # 2. 批量文本纠错
    # batch_texts = [
    #     "感帽发烧",
    #     "咳簌不止",
    #     "头通恶心",
    #     "呼吸困難",  # 故意输入繁体/错字
    #     "今天天氣不好，我感帽了"
    # ]
    # batch_results = text_correction_predict(
    #     text=batch_texts,
    #     model_dir=BEST_MODEL_DIR,
    #     batch_size=3,  # 批次大小设为3
    #     num_beams=5
    # )
    # print("\n" + "=" * 60)
    # print("【批量文本纠错结果】")
    # for idx, (origin, corrected) in enumerate(zip(batch_texts, batch_results), 1):
    #     print(f"{idx}. 原始：{origin} → 纠错：{corrected}")
    # print("=" * 60)