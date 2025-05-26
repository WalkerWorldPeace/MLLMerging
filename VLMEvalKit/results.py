# -*- coding: gbk -*-
import os
import sys
import csv

def process_mathvista_content(content):
    target_names = {"geometry reasoning", "algebraic reasoning", "geometry problem solving"}
    values = []
    try:
        reader = csv.reader(content.splitlines())
        headers = next(reader, None)  # ����б�ͷ�������������ע�ʹ���
        for row in reader:
            if row and row[0].strip().lower() in target_names:
                try:
                    value = float(row[-1].strip().strip('"'))
                    values.append(value)
                except Exception:
                    pass
    except Exception:
        pass
    if values:
        return sum(values) / len(values)
    return None

def process_mathvision_content(content):
    target_names = {"metric geometry - angle", "metric geometry - area", "metric geometry - length", "solid geometry"}
    values = []
    try:
        reader = csv.reader(content.splitlines())
        # ��� CSV �ļ��б�ͷ����ȡ����һ��ע��
        # headers = next(reader, None)
        for row in reader:
            if row and row[0].strip().lower() in target_names:
                try:
                    value = float(row[-1].strip().strip('"'))
                    values.append(value)
                except Exception:
                    pass
    except Exception:
        pass
    if values:
        return sum(values) / len(values)
    return None

def search_and_print(folder):
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if 'acc' in filename.lower() or 'score' in filename.lower():
                file_path = os.path.join(root, filename)
                if os.path.islink(file_path):
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    content = f'��ȡ�ļ�ʱ������{e}'
                print(f'�ļ���: {file_path}')
                print('����:')
                print(content)
                # ����ļ������� "MathVista"������ָ���������һ����ֵ��ƽ��ֵ
                if 'mathvista' in filename.lower():
                    avg = process_mathvista_content(content)
                    if avg is not None:
                        print(f'ָ���������һ����ֵƽ��ֵ: {avg}')
                    else:
                        print('δ���ҵ�ָ�����л����ʧ��')
                # ����ļ������� "MathVision"������ָ���������һ����ֵ��ƽ��ֵ
                if 'mathvision' in filename.lower():
                    avg = process_mathvision_content(content)
                    if avg is not None:
                        print(f'ָ���������һ����ֵƽ��ֵ: {avg}')
                    else:
                        print('δ���ҵ�ָ�����л����ʧ��')
                print('-' * 50)

if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    search_and_print(folder)