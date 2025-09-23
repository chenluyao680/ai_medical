import json


def convert_to_doccano(input_data):
    """
    转换包含多个意图标签的数据为doccano格式

    参数:
        input_data: 原始数据字典，可能包含多个intent

    返回:
        doccano格式的字典
    """
    # 基础格式包含文本和标签列表
    doccano_data = {
        "text": input_data["text"],
        "label": []  # 初始化为空列表，用于收集所有标签
    }

    # 添加意图标签
    if input_data.get("intent"):
        doccano_data["label"].extend(input_data["intent"])

    # 添加request信息，格式化为"request:内容"
    if input_data.get("request"):
        doccano_data["label"].extend([f"request:{item}" for item in input_data["request"]])

    # 添加consult信息，格式化为"consult:内容"
    if input_data.get("consult"):
        doccano_data["label"].extend([f"consult:{item}" for item in input_data["consult"]])

    return doccano_data


# 示例数据（包含多个intent）
sample_data = {
    "text": "查询糖尿病的胰岛素注射技巧，并反馈挂号系统故障问题",
    "intent": ["request", "consult"],
    "request": ["系统故障反馈"],
    "consult": ["疾病对应详情"]
}

# 转换示例
doccano_format = convert_to_doccano(sample_data)
print(json.dumps(doccano_format, ensure_ascii=False, indent=2))


# 批量转换函数（处理JSONL文件）
def batch_convert(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            try:
                data = json.loads(line.strip())
                converted = convert_to_doccano(data)
                f_out.write(json.dumps(converted, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"处理数据时出错: {e}, 数据行: {line}")

# 使用示例：
batch_convert("D:\\python_project\\test\\ai_medical\\data\\intent_classify\\raw\\data.jsonl", "D:\\python_project\\test\\ai_medical\\data\\intent_classify\\raw\\output_doccano.jsonl")