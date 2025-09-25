import json


def convert_cmeie_to_doccano(cmeie_line):
    """
    将单条CMeIE-V2格式数据转换为doccano关系抽取格式
    :param cmeie_line: 单条CMeIE数据（dict类型）
    :return: 转换后的doccano格式数据（dict类型）
    """
    text = cmeie_line["text"]
    spo_list = cmeie_line["spo_list"]

    # 1. 收集所有实体（主语+宾语），去重并记录类型，避免重复标注
    entities = []  # 存储格式：(实体文本, 实体类型)
    for spo in spo_list:
        # 处理主语
        subj_text = spo["subject"]
        subj_type = spo["subject_type"]
        if (subj_text, subj_type) not in entities:
            entities.append((subj_text, subj_type))
        # 处理宾语（CMeIE中宾语是{"@value": "..."}，类型是{"@value": "..."}）
        obj_text = spo["object"]["@value"]
        obj_type = spo["object_type"]["@value"]
        if (obj_text, obj_type) not in entities:
            entities.append((obj_text, obj_type))

    # 2. 构建annotations（实体标注列表）：含id、start、end、label、text
    annotations = []
    entity_id_map = {}  # 映射：(实体文本, 实体类型) -> 实体ID，用于后续关联关系
    for idx, (ent_text, ent_type) in enumerate(entities):
        # 计算实体在text中的位置（默认取第一次出现的位置）
        start = text.find(ent_text)
        if start == -1:
            # 极端情况：实体未在文本中找到（可能数据问题），跳过该实体
            print(f"警告：实体'{ent_text}'未在文本中找到，已跳过")
            continue
        end = start + len(ent_text)
        # 生成标注项
        annotation = {
            "id": idx,
            "start_offset": start,
            "end_offset": end,
            "label": ent_type,
            "text": ent_text
        }
        annotations.append(annotation)
        entity_id_map[(ent_text, ent_type)] = idx

    # 3. 构建relations（关系列表）：含subject_id、object_id、predicate
    relations = []
    for spo in spo_list:
        # 获取主语ID
        subj_text = spo["subject"]
        subj_type = spo["subject_type"]
        subj_id = entity_id_map.get((subj_text, subj_type))
        if subj_id is None:
            continue  # 跳过未找到的实体（数据问题）
        # 获取宾语ID
        obj_text = spo["object"]["@value"]
        obj_type = spo["object_type"]["@value"]
        obj_id = entity_id_map.get((obj_text, obj_type))
        if obj_id is None:
            continue  # 跳过未找到的实体（数据问题）
        # 生成关系项
        relation = {
            "subject_id": subj_id,
            "object_id": obj_id,
            "predicate": spo["predicate"]
        }
        relations.append(relation)

    # 4. 组装最终的doccano格式
    doccano_data = {
        "text": text,
        "annotations": annotations,
        "relations": relations
    }
    return doccano_data


# ------------------- 批量处理JSONL文件 -------------------
# 输入文件路径（CMeIE-V2.jsonl）
input_path = "D:\\python_project\\test\\ai_medical\\data\\annotated_data\\CMeIE-V2.jsonl"
# 输出文件路径（转换后的doccano格式）
output_path = "./CMeIE-V2_doccano.jsonl"

with open(input_path, "r", encoding="utf-8") as f_in, \
        open(output_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        # 读取单条CMeIE数据
        cmeie_data = json.loads(line.strip())
        # 转换格式
        doccano_data = convert_cmeie_to_doccano(cmeie_data)
        # 写入输出文件（每行一个JSON）
        f_out.write(json.dumps(doccano_data, ensure_ascii=False) + "\n")

print(f"转换完成！输出文件已保存至：{output_path}")