import json
import os
import re
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")


# -------------------------- 拼写检查功能 --------------------------
def text_correction_predict(
        text: str | list[str],
        model_dir: str,
        max_length: int = 128,
        batch_size: int = 8,
        num_beams: int = 5
) -> str | list[str]:
    """文本纠错预测函数，支持单条/批量文本纠错"""
    # 自动选择设备（优先GPU，无GPU则用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和Tokenizer（关键修复：确保路径正确识别为本地路径）
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()  # 切换为评估模式

    # 统一输入格式（单条→列表）
    is_single_input = isinstance(text, str)
    if is_single_input:
        text_list = [text.strip()]
        if not text_list[0]:
            raise ValueError("待纠错文本不能为空！")
    else:
        text_list = [t.strip() for t in text if isinstance(t, str) and t.strip()]
        if not text_list:
            raise ValueError("批量待纠错文本不能为空！")

    # 生成参数配置（不含device参数）
    generation_config = {
        "max_length": max_length,
        "min_length": 1,
        "num_beams": num_beams,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
        "temperature": 1.0,
        "top_p": 1.0
    }

    # 批量预测
    corrected_results = []
    for i in range(0, len(text_list), batch_size):
        batch_text = text_list[i:i + batch_size]

        # 文本编码
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

        # 模型预测
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, **generation_config
            )

        # 解码结果
        batch_corrected = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        corrected_results.extend(batch_corrected)

    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return corrected_results[0] if is_single_input else corrected_results


def clean_text(text: str) -> str:
    """去除文本中的意外字符和多余空格"""
    # 定义允许的字符模式
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9,.，。:：;；()（）%％-]'
    # 替换不允许的字符为空格
    cleaned = re.sub(pattern, ' ', text)
    # 合并多个空格为一个
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# -------------------------- 核心转换功能 --------------------------
def find_all_occurrences(text: str, target: str) -> List[Tuple[int, int]]:
    """查找目标字符串在文本中所有出现的位置（开始和结束索引）"""
    occurrences = []
    start = 0
    target_len = len(target)
    text_len = len(text)

    while start <= text_len - target_len:
        pos = text.find(target, start)
        if pos == -1:
            break
        # 记录起始和结束索引（Doccano的end是闭区间）
        occurrences.append((pos, pos + target_len - 1))
        start = pos + 1

    return occurrences


def convert_spo_to_doccano(item: Dict, correction_model_dir: str) -> Dict:
    """将单条SPO格式数据转换为Doccano格式，包含拼写检查和文本清理"""
    # 1. 清理文本，去除意外字符
    raw_text = item["text"]
    cleaned_text = clean_text(raw_text)

    # 2. 对清理后的文本进行拼写检查
    corrected_text = text_correction_predict(
        text=cleaned_text,
        model_dir=correction_model_dir,
        num_beams=5
    )

    spo_list = item["spo_list"]

    # 存储所有实体（去重）
    entities = []
    entity_id_map = {}  # 用于去重：(实体文本, 实体类型) -> 实体ID
    current_entity_id = 0

    # 存储所有关系
    relations = []
    current_relation_id = 0

    # 3. 提取所有实体并去重（基于纠错后的文本）
    for spo in spo_list:
        # 处理主语
        subject = spo["subject"]
        subject_type = spo["subject_type"]
        subject_key = (subject, subject_type)

        if subject_key not in entity_id_map:
            # 查找主语在纠错后文本中的所有位置
            subject_occurrences = find_all_occurrences(corrected_text, subject)
            if subject_occurrences:
                # 取第一个出现的位置作为标注位置
                start, end = subject_occurrences[0]
                entities.append({
                    "id": current_entity_id,
                    "start_offset": start,
                    "end_offset": end,
                    "label": subject_type
                })
                entity_id_map[subject_key] = current_entity_id
                current_entity_id += 1

        # 处理宾语
        object_value = spo["object"]["@value"]
        object_type = spo["object_type"]["@value"]
        object_key = (object_value, object_type)

        if object_key not in entity_id_map:
            # 查找宾语在纠错后文本中的所有位置
            object_occurrences = find_all_occurrences(corrected_text, object_value)
            if object_occurrences:
                # 取第一个出现的位置作为标注位置
                start, end = object_occurrences[0]
                entities.append({
                    "id": current_entity_id,
                    "start_offset": start,
                    "end_offset": end,
                    "label": object_type
                })
                entity_id_map[object_key] = current_entity_id
                current_entity_id += 1

    # 4. 提取所有关系
    for spo in spo_list:
        subject = spo["subject"]
        subject_type = spo["subject_type"]
        subject_key = (subject, subject_type)

        object_value = spo["object"]["@value"]
        object_type = spo["object_type"]["@value"]
        object_key = (object_value, object_type)

        # 确保主语和宾语都已在实体列表中
        if subject_key in entity_id_map and object_key in entity_id_map:
            relations.append({
                "id": current_relation_id,
                "from_id": entity_id_map[subject_key],
                "to_id": entity_id_map[object_key],
                "type": spo["predicate"]
            })
            current_relation_id += 1

    return {
        "text": corrected_text,  # 使用纠错后的文本
        "original_text": raw_text,  # 保留原始文本用于参考
        "entities": entities,
        "relations": relations
    }


def batch_convert(input_path: str, output_file: str, correction_model_dir: str) -> None:
    """批量转换文件（支持单个文件或目录下所有文件）为Doccano格式"""
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 收集需要处理的所有文件路径
    files_to_process = []
    if os.path.isdir(input_path):
        # 如果是目录，处理目录下所有JSON/JSONL文件
        for filename in os.listdir(input_path):
            if filename.endswith((".json", ".jsonl")):
                files_to_process.append(os.path.join(input_path, filename))
    elif os.path.isfile(input_path) and input_path.endswith((".json", ".jsonl")):
        # 如果是单个文件，直接加入处理列表
        files_to_process.append(input_path)
    else:
        raise ValueError(f"无效的输入路径：{input_path}（必须是JSON/JSONL文件或包含它们的目录）")

    # 处理所有收集到的文件
    with open(output_file, "w", encoding="utf-8") as out_f:
        for file_path in files_to_process:
            print(f"处理文件: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    # 处理JSONL文件（每行一个JSON对象）
                    if file_path.endswith(".jsonl"):
                        items = []
                        for line in in_f:
                            line = line.strip()
                            if line:
                                items.append(json.loads(line))
                    # 处理普通JSON文件（数组或单个对象）
                    else:
                        data = json.load(in_f)
                        items = data if isinstance(data, list) else [data]

                    # 逐条转换并写入
                    for item in items:
                        if "text" in item and "spo_list" in item:
                            doccano_item = convert_spo_to_doccano(item, correction_model_dir)
                            out_f.write(json.dumps(doccano_item, ensure_ascii=False) + "\n")

                print(f"完成处理: {os.path.basename(file_path)}，生成{len(items)}条数据")

            except Exception as e:
                print(f"处理文件{file_path}出错: {str(e)}")

    print(f"所有文件处理完成，输出至: {output_file}")


if __name__ == "__main__":
    # 配置路径 - 关键修复：在模型路径前加r表示原始字符串
    INPUT_PATH = r"D:\python_project\test\ai_medical\data\annotated_data\CMeIE-V2.jsonl"
    OUTPUT_FILE = r"D:\python_project\test\ai_medical\data\annotated_data\doccano_annotations.jsonl"
    # 关键修复：模型路径使用原始字符串格式（加r前缀）
    CORRECTION_MODEL_DIR = "D:\\python_project\\test\\ai_medical\\models\\bart_correction\\best_model_epoch1_valloss0.7263"

    # 执行批量转换
    batch_convert(INPUT_PATH, OUTPUT_FILE, CORRECTION_MODEL_DIR)
