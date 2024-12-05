import json
import os

dir_path = r"D:\files\PHD\myNeRF\nerfstudio\data\Semantic"

os.path.dirname(dir_path)
json_file_path = dir_path + r"\transforms.json"

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for frame in data.get("frames", []):
    if "file_path" in frame:
        frame["file_path"] = frame["file_path"].replace(r"data/Semantic", r".")
# 将修改后的数据写回文件
with open(dir_path + r"\transforms.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("file_path 已成功加上前缀。")
