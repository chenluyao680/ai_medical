import os
import json
import csv
from typing import List, Dict, Callable
from tqdm import tqdm  # 进度条库，可选（需安装：pip install tqdm）

# -------------------------- 1. 配置参数（可根据需求修改） --------------------------
# 1.1 符号过滤配置：需去除的多余符号（可按需添加，如@、#、*等）
REMOVE_SYMBOLS = {'@', '*', '#', '$', '%', '^', '&', '!', '?', ';', ':', '"', "'"}
# 1.2 数据路径配置
INPUT_DIR = "./input_data"    # 输入数据文件夹（存放待处理的JSON/CSV/TXT文件）
OUTPUT_DIR = "./output_data"  # 输出数据文件夹（存放处理后的文件）
# 1.3 文本字段配置：不同数据格式中“文本字段”的键名（如JSON的"text"，CSV的"content"）
TEXT_FIELD_MAP = {
    "json": "text",    # JSON文件中，文本内容的字段名（如你的数据中的"text"字段）
    "csv": "content",  # CSV文件中，文本内容的字段名（需与你的CSV表头一致）
    "txt": "text"      # TXT文件无需字段名，直接读取每行文本
}


# -------------------------- 2. 核心工具函数 --------------------------
def spell_check_dummy(text: str) -> str:
    """
    【拼写检查示例函数】：请替换为你的真实spellCheck函数
    功能：模拟拼写检查（此处仅做简单修正，实际需替换为你的模型/工具实现）
    :param text: 待纠错文本
    :return: 纠错后文本
    """
    # 示例修正规则（替换为你的真实spellCheck逻辑，如调用BERT/BART模型、第三方工具）
    correction_map = {
        "溶血性贫_血": "溶血性贫血",
        "系统性红斑琅疮": "系统性红斑狼疮",
        "自身抗_体": "自身抗体"
    }
    corrected_text = text
    for wrong, right in correction_map.items():
        corrected_text = corrected_text.replace(wrong, right)
    return corrected_text


def remove_excess_symbols(text: str, symbols: set = REMOVE_SYMBOLS) -> str:
    """
    去除文本中的多余符号（如@、*、#等）
    :param text: 待处理文本
    :param symbols: 需去除的符号集合（默认使用配置中的REMOVE_SYMBOLS）
    :return: 去除符号后的干净文本
    """
    cleaned_text = text
    for symbol in symbols:
        cleaned_text = cleaned_text.replace(symbol, "")
    # 额外处理：去除多余空格（将连续空格转为单个空格）
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text


# -------------------------- 3. 批量数据读取函数（支持多格式） --------------------------
def read_json_file(file_path: str, text_field: str = TEXT_FIELD_MAP["json"]) -> List[Dict]:
    """读取JSON文件（支持单条JSON对象或JSON数组）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 若JSON是单条对象（如{"text": "...", "spo_list": [...]}），转为列表
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"JSON文件格式错误：既非对象也非数组，路径：{file_path}")
    except Exception as e:
        print(f"读取JSON文件失败（{file_path}）：{str(e)}")
        return []


# 格式-读取器映射：新增格式时，只需添加此处的映射和对应的读取函数
FILE_READER_MAP = {
    "json": read_json_file
}


def read_batch_files(input_dir: str) -> Dict[str, List[Dict]]:
    """
    批量读取输入文件夹下的所有支持格式文件
    :param input_dir: 输入文件夹路径
    :return: 字典{文件名: 数据列表}
    """
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"输入文件夹不存在，已自动创建：{input_dir}")
        return {}

    batch_data = {}
    # 遍历文件夹下所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        # 跳过文件夹（只处理文件）
        if os.path.isdir(file_path):
            continue
        # 获取文件格式（后缀名）
        file_ext = filename.split(".")[-1].lower()
        if file_ext not in FILE_READER_MAP:
            print(f"跳过不支持的文件格式：{filename}（支持格式：{list(FILE_READER_MAP.keys())}）")
            continue
        # 调用对应格式的读取器
        reader = FILE_READER_MAP[file_ext]
        data = reader(file_path)
        if data:
            batch_data[filename] = data
            print(f"成功读取文件：{filename}（数据条数：{len(data)}）")
        else:
            print(f"文件无有效数据：{filename}")
    return batch_data


# -------------------------- 4. 批量数据处理函数（拼写检查+符号去除） --------------------------
def process_batch_data(batch_data: Dict[str, List[Dict]], spell_check_func: Callable) -> Dict[str, List[Dict]]:
    """
    批量处理数据：对每条文本先执行拼写检查，再去除多余符号
    :param batch_data: 批量读取的数据{文件名: 数据列表}
    :param spell_check_func: 拼写检查函数（需传入你的真实spellCheck函数）
    :return: 处理后的批量数据
    """
    processed_batch = {}
    # 遍历每个文件的数据
    for filename, data_list in batch_data.items():
        processed_data = []
        # 用进度条显示处理进度（可选，需安装tqdm）
        for item in tqdm(data_list, desc=f"处理文件：{filename}"):
            try:
                # 1. 获取当前文本（根据文件格式确定文本字段）
                file_ext = filename.split(".")[-1].lower()
                text_field = TEXT_FIELD_MAP[file_ext]
                original_text = item.get(text_field, "")
                if not original_text:
                    processed_data.append({**item, f"{text_field}_processed": "", "process_status": "跳过（空文本）"})
                    continue

                # 2. 第一步：执行拼写检查
                corrected_text = spell_check_func(original_text)

                # 3. 第二步：去除多余符号
                cleaned_text = remove_excess_symbols(corrected_text)

                # 4. 保存处理结果（保留原始数据，新增处理后的字段）
                processed_item = {
                    **item,  # 保留原始所有字段（如spo_list、line_number等）
                    f"{text_field}_original": original_text,  # 原始文本（便于对比）
                    f"{text_field}_corrected": corrected_text,  # 拼写检查后文本
                    f"{text_field}_processed": cleaned_text,   # 最终处理文本（纠错+去符号）
                    "process_status": "成功"
                }
                processed_data.append(processed_item)

            except Exception as e:
                # 处理单条数据失败时，记录错误状态，不中断整体流程
                error_item = {**item, f"{text_field}_processed": "", "process_status": f"失败：{str(e)}"}
                processed_data.append(error_item)
                print(f"处理单条数据失败（文件：{filename}）：{str(e)}")

        processed_batch[filename] = processed_data
        print(f"文件处理完成：{filename}（成功：{len([x for x in processed_data if x['process_status'] == '成功'])}条）")
    return processed_batch


# -------------------------- 5. 批量数据保存函数（支持多格式，与输入格式对应） --------------------------
def save_json_file(data: List[Dict], output_path: str):
    """保存为JSON文件"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON文件保存成功：{output_path}")
    except Exception as e:
        print(f"JSON文件保存失败（{output_path}）：{str(e)}")



def save_batch_data(processed_batch: Dict[str, List[Dict]], output_dir: str):
    """
    批量保存处理后的数据（与输入文件格式对应，保存到output_dir）
    :param processed_batch: 处理后的批量数据{文件名: 数据列表}
    :param output_dir: 输出文件夹路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"输出文件夹不存在，已自动创建：{output_dir}")

    for filename, data in processed_batch.items():
        # 生成输出路径（保持原文件名，添加"_processed"后缀）
        name_part, ext_part = ".".join(filename.split(".")[:-1]), filename.split(".")[-1].lower()
        output_filename = f"{name_part}_processed.{ext_part}"
        output_path = os.path.join(output_dir, output_filename)

        # 调用对应格式的写入器
        if ext_part in FILE_WRITER_MAP:
            writer = FILE_WRITER_MAP[ext_part]
            writer(data, output_path)
        else:
            print(f"跳过不支持的保存格式：{filename}")


# -------------------------- 6. 主函数：一键执行批量处理 --------------------------
def main(spell_check_func: Callable = spell_check_dummy):
    """
    主函数：批量读取→批量处理→批量保存
    :param spell_check_func: 你的拼写检查函数（默认使用示例函数，需替换为真实函数）
    """
    print("=" * 50)
    print("开始批量文本处理（拼写检查+多余符号去除）")
    print("=" * 50)

    # 1. 批量读取输入文件
    print("\n【步骤1：读取输入文件】")
    batch_data = read_batch_files(INPUT_DIR)
    if not batch_data:
        print("无有效输入文件，流程终止")
        return

    # 2. 批量处理数据（拼写检查+去符号）
    print("\n【步骤2：处理数据】")
    processed_batch = process_batch_data(batch_data, spell_check_func)

    # 3. 批量保存处理结果
    print("\n【步骤3：保存结果】")
    save_batch_data(processed_batch, OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("批量处理流程全部完成！")
    print(f"处理结果保存路径：{os.path.abspath(OUTPUT_DIR)}")
    print("=" * 50)


# -------------------------- 7. 执行入口（替换拼写检查函数后运行） --------------------------
if __name__ == "__main__":
    # ！！！关键：将spell_check_dummy替换为你的真实spellCheck函数
    # 示例：若你的拼写检查函数名为my_real_spell_check，则改为 main(spell_check_func=my_real_spell_check)
    main(spell_check_func=spell_check_dummy)