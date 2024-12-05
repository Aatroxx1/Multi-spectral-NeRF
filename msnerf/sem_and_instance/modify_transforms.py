import os
import json
import shutil


def transform_filepath(folder, new_path):
    """
    修改transforms.json中file_path的路径前缀

    Args:
        folder: transforms.json所在的文件夹路径
        new_path: 新的路径前缀，例如 "data/Semantic/images/"
    """
    transforms_path = os.path.join(folder, 'transforms.json')

    # 检查源文件是否存在
    if not os.path.exists(transforms_path):
        print(f"File '{transforms_path}' does not exist. Skipping operation.")
        return

    # 备份原始文件
    backup_path = os.path.join(folder, 'transforms_backup.json')
    shutil.copyfile(transforms_path, backup_path)

    with open(transforms_path, 'r') as f:
        data = json.load(f)

    # 修改frames列表中的file_path字段
    if 'frames' in data:
        for frame in data['frames']:
            if 'file_path' in frame:
                # 获取原始文件名
                original_filename = os.path.basename(frame['file_path'])
                # 组合新路径
                frame['file_path'] = os.path.join(new_path, original_filename)
                # 将Windows路径分隔符替换为正斜杠
                frame['file_path'] = frame['file_path'].replace('\\', '/')

    # 将修改后的数据写回transforms.json文件
    with open(transforms_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    folder = r"D:\files\PHD\myNeRF\nerfstudio\data\Semantic"
    new_path = "data/Semantic/images"
    transform_filepath(folder, new_path)