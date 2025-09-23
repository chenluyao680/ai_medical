import os
import random
import json
from sklearn.model_selection import train_test_split

# -------------------------- 配置参数（可按需修改） --------------------------
# 原始数据目录：存放以“同一行空格分隔错误句和正确句”的TXT文件
RAW_DATA_DIR = "D:\\python_project\\test\\ai_medical\\data\\spell_check\\raw"
# 处理后数据目录：保存训练/验证/测试集（JSON格式，便于后续训练读取）
PROCESSED_DATA_DIR = "D:\\python_project\\test\\ai_medical\\data\\spell_check\\processed"
# 数据分割比例：训练集80%，验证集10%，测试集10%
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
# 随机种子：保证数据分割结果可复现
RANDOM_SEED = 42


def clean_text(text: str) -> str:
    """
    文本清洗：去除多余符号、空格，统一格式
    :param text: 原始文本（错误句或正确句）
    :return: 清洗后的文本
    """
    if not text:
        return ""
    # 1. 去除首尾空白字符（换行符、空格、制表符）
    cleaned = text.strip("\n\r\t ")
    # 2. 去除多余符号（可根据实际数据补充，如特殊符号、乱码字符）
    excess_symbols = {"@", "*", "#", "$", "%", "^", "&", "!", "?", ";", ":", "\"", "'", "`"}
    for symbol in excess_symbols:
        cleaned = cleaned.replace(symbol, "")
    # 3. 统一空格：将连续空格转为单个空格（避免句子内部多空格）
    cleaned = " ".join(cleaned.split())
    # 4. 修复明显的编码/格式错误（如“子灾难”→“在灾难”，根据你的数据补充常见错误）
    common_fixes = {
        "子灾难": "在灾难",  # 你的示例数据中“子灾难”是错误，正确为“在灾难”
        "候这么短": "后这么短"  # 你的示例数据中“候这么短”是错误，正确为“后这么短”
    }
    for wrong, right in common_fixes.items():
        cleaned = cleaned.replace(wrong, right)
    return cleaned


def load_data() -> list[dict]:
    """
    加载原始数据：解析“同一行空格分隔错误句和正确句”的TXT文件
    :return: 数据列表，每个元素为{"input": 错误句, "target": 正确句}
    """
    data = []
    # 检查原始数据目录是否存在
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"原始数据目录不存在：{RAW_DATA_DIR}，请先创建并放入TXT文件")

    # 遍历目录下所有TXT文件
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(".txt"):
            print(f"跳过非TXT文件：{filename}")
            continue

        file_path = os.path.join(RAW_DATA_DIR, filename)
        print(f"正在读取文件：{file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            # 按行读取文件（每行是一个样本：错误句 正确句）
            for line_num, line in enumerate(f, 1):
                try:
                    # 关键：按“第一个空格”分割错误句和正确句（避免句子内部空格干扰）
                    # split(" ", 1) 表示只分割一次，得到 [错误句, 正确句]
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        print(f"警告：第{line_num}行格式错误（未找到空格分隔的两组数据），跳过该行：{line.strip()}")
                        continue

                    # 提取错误句和正确句，并清洗
                    error_sentence = clean_text(parts[0])
                    correct_sentence = clean_text(parts[1])

                    # 跳过清洗后为空的样本
                    if not error_sentence or not correct_sentence:
                        print(f"警告：第{line_num}行清洗后为空，跳过该行")
                        continue

                    # 添加到数据列表
                    data.append({
                        "input": error_sentence,
                        "target": correct_sentence,
                        "source_file": filename,  # 记录来源文件，便于后续溯源
                        "line_number": line_num  # 记录行号，便于后续排查错误
                    })

                except Exception as e:
                    print(f"错误：处理第{line_num}行时出错，跳过该行：{str(e)}")
                    continue

    # 检查是否加载到数据
    if not data:
        raise ValueError("未加载到任何有效数据，请检查TXT文件格式是否为“错误句 正确句”")

    print(f"成功加载 {len(data)} 条有效样本")
    return data


def split_data(data: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    分割数据为训练集、验证集、测试集
    :param data: 完整数据列表
    :return: (训练集, 验证集, 测试集)
    """
    # 第一步：分割训练集和“验证集+测试集”（8:2）
    train_data, temp_data = train_test_split(
        data,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True  # 打乱数据，保证分布均匀
    )
    # 第二步：分割验证集和测试集（1:1，即总数据的10%:10%）
    val_data, test_data = train_test_split(
        temp_data,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print(f"数据分割完成：")
    print(f"- 训练集：{len(train_data)} 条（{TRAIN_RATIO * 100:.1f}%）")
    print(f"- 验证集：{len(val_data)} 条（{VAL_RATIO * 100:.1f}%）")
    print(f"- 测试集：{len(test_data)} 条（{TEST_RATIO * 100:.1f}%）")
    return train_data, val_data, test_data


def save_data(train_data: list[dict], val_data: list[dict], test_data: list[dict]):
    """
    保存处理后的数据到JSON文件（便于train.py读取）
    :param train_data: 训练集
    :param val_data: 验证集
    :param test_data: 测试集
    """
    # 创建处理后数据目录（若不存在）
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 定义保存路径
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.json")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.json")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.json")

    # 保存JSON文件（ensure_ascii=False 支持中文，indent=2 格式化显示）
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"处理后数据已保存到：{PROCESSED_DATA_DIR}")
    print(f"- 训练集：{train_path}")
    print(f"- 验证集：{val_path}")
    print(f"- 测试集：{test_path}")


def main():
    """主函数：加载→清洗→分割→保存数据"""
    print("=" * 60)
    print("开始数据处理流程（解析“同一行空格分隔”的纠错数据）")
    print("=" * 60)

    try:
        # 1. 加载原始数据
        print("\n【步骤1/4】加载原始数据...")
        data = load_data()

        # 2. 分割数据
        print("\n【步骤2/4】分割训练集/验证集/测试集...")
        train_data, val_data, test_data = split_data(data)

        # 3. 保存处理后数据
        print("\n【步骤3/4】保存处理后数据...")
        save_data(train_data, val_data, test_data)

        # 4. 输出处理总结
        print("\n【步骤4/4】数据处理完成！")
        print(f"总样本数：{len(data)} 条")
        print(f"训练集：{len(train_data)} 条 | 验证集：{len(val_data)} 条 | 测试集：{len(test_data)} 条")
        print(f"处理后数据路径：{os.path.abspath(PROCESSED_DATA_DIR)}")
        print("=" * 60)

    except Exception as e:
        print(f"\n数据处理失败：{str(e)}")
        print("=" * 60)


if __name__ == "__main__":
    main()