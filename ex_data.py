import xml.etree.ElementTree as ET
import json
import os
import random  # 用于随机打乱


def extract_real_dti_fixed(xml_file_path, base_output_json_path):
    ns = {'db': 'http://www.drugbank.ca'}
    all_records = []

    pos_count = 0
    neg_count = 0

    print(f"⏳ 正在解析真正的 DrugBank 数据库: {xml_file_path} ...")

    context = ET.iterparse(xml_file_path, events=('start', 'end'))
    context_iter = iter(context)
    event, root = next(context_iter)

    for event, elem in context_iter:
        if event == 'end' and elem.tag.endswith('drug'):

            smiles = None
            calc_props = elem.find('db:calculated-properties', ns)
            if calc_props is not None:
                for prop in calc_props.findall('db:property', ns):
                    kind = prop.find('db:kind', ns)
                    if kind is not None and kind.text == 'SMILES':
                        val = prop.find('db:value', ns)
                        if val is not None:
                            smiles = val.text
                            break

            if smiles:
                name_elem = elem.find('db:name', ns)
                drug_name = name_elem.text if name_elem is not None else "Unknown_Drug"

                targets_elem = elem.find('db:targets', ns)
                if targets_elem is not None:
                    for target in targets_elem.findall('db:target', ns):

                        actions_elem = target.find('db:actions', ns)
                        action_types = []
                        if actions_elem is not None:
                            for act in actions_elem.findall('db:action', ns):
                                if act.text:
                                    action_types.append(act.text.lower())

                        if any(a in action_types for a in
                               ['inhibitor', 'antagonist', 'blocker', 'suppressor', 'down-regulator']):
                            label = 0
                            action_desc = "inhibits/blocks"
                        elif any(a in action_types for a in
                                 ['agonist', 'activator', 'inducer', 'stimulator', 'up-regulator']):
                            label = 1
                            action_desc = "activates/stimulates"
                        else:
                            label = 1
                            action_desc = "binds to"

                        polypeptide = target.find('db:polypeptide', ns)
                        if polypeptide is not None:
                            sequence_elem = polypeptide.find('db:amino-acid-sequence', ns)

                            if sequence_elem is not None and sequence_elem.text:
                                target_name_elem = target.find('db:name', ns)
                                target_name = target_name_elem.text if target_name_elem is not None else "Unknown_Target"

                                sequence = sequence_elem.text.replace('\n', '').replace('\r', '').replace(' ', '')

                                record = {
                                    "drug_smiles": smiles,
                                    "target_sequence": sequence,
                                    "label": label,
                                    "explanation": f"{drug_name} {action_desc} {target_name}.",
                                    "exp_id": 1
                                }
                                all_records.append(record)

                                if label == 1:
                                    pos_count += 1
                                else:
                                    neg_count += 1

            root.clear()

    print(f"\n✅ 解析完成！这次共提取到了 {len(all_records)} 条真实的药物-靶标互作数据。")
    print(f"   📈 正向数据 (Label=1，激动/激活/结合): {pos_count} 条")
    print(f"   📉 负向数据 (Label=0，抑制/拮抗/阻断): {neg_count} 条\n")

    # ================= 核心修改部分 =================
    # 循环 4 次，生成 4 个包含全部数据但随机打乱的 json 文件
    for i in range(1, 5):
        # 原地随机打乱列表所有元素
        random.shuffle(all_records)

        # 构造带有编号的文件名
        output_json_path = base_output_json_path.replace('.json', f'_{i}.json')

        # 将打乱后的数据写入当前编号的文件
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(all_records, f_out, indent=4, ensure_ascii=False)

        print(f"🎉 文件 {i}/4 已打乱并保存！文件路径: {output_json_path}")
    # =================================================


if __name__ == "__main__":
    xml_file = "full database.xml"
    output_file = "drugbank_dti.json"

    if os.path.exists(xml_file):
        extract_real_dti_fixed(xml_file, output_file)
    else:
        print(f"❌ 找不到文件 {xml_file}，请检查路径。")