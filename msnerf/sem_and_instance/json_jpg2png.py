import os
import json
import shutil
import re

def transformjson_jpg_to_png(dataset_root):
    transforms_path = os.path.join(dataset_root, 'transforms.json')

    # 检查源文件是否存在
    if not os.path.exists(transforms_path):
        print(f"File '{transforms_path}' does not exist. Skipping operation.")
        return

    # 备份原始文件
    backup_path = os.path.join(dataset_root, 'transforms_jpg.json')
    shutil.copyfile(transforms_path, backup_path)

    with open(transforms_path, 'r') as f:
        data = json.load(f)

    # 修改frames列表中的file_path字段
    if 'frames' in data:
        for frame in data['frames']:
            if 'file_path' in frame and frame['file_path'].lower().endswith('.jpg'):
                # 使用正则表达式进行大小写不敏感的替换
                frame['file_path'] = re.sub(r'(?i)\.jpg$', '.png', frame['file_path'])

    # 将修改后的数据写回transforms.json文件
    with open(transforms_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    folder = r"D:\files\PHD\myNeRF\nerfstudio\data\Semantic"
    transformjson_jpg_to_png(folder)