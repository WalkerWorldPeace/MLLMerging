import json
import os
import sys
import re

def convert_box(box):
    """将[x_min, y_min, x_max, y_max]格式的边界框转换为特定格式的字符串"""
    # 将坐标映射到[0,1000)范围
    x_min = int(box[0] * 1000)
    y_min = int(box[1] * 1000)
    x_max = int(box[2] * 1000)
    y_max = int(box[3] * 1000)
    
    # 转换为特定格式
    return f"<|box_start|>({x_min},{y_min}),({x_max},{y_max})<|box_end|>"

def process_value_string(value):
    """处理value字符串中的边界框表示"""
    # 正则表达式匹配形如[0.47, 0.07, 0.97, 0.92]的边界框
    pattern = r'\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]'
    
    def replace_match(match):
        box = [float(match.group(i)) for i in range(1, 5)]
        return convert_box(box)
    
    # 替换所有匹配的边界框
    return re.sub(pattern, replace_match, value)

def process_item(item):
    """处理单个样本中的conversations"""
    if "conversations" in item:
        for conv in item["conversations"]:
            if "value" in conv and isinstance(conv["value"], str):
                conv["value"] = process_value_string(conv["value"])

def main():
    if len(sys.argv) < 2:
        print("使用方法: python grounding.py <input_json_file> [output_json_file]")
        return
    
    input_file = sys.argv[1]
    
    # 如果没有提供输出文件名，则基于输入文件名生成
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.json"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 根据数据结构进行不同的处理
        if isinstance(data, list):
            # 顶级是列表，每个项目都是有conversations的字典
            for item in data:
                process_item(item)
        else:
            # 假设是单一对象
            process_item(data)
        
        # 保存转换后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成。结果保存至 {output_file}")
    
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main()