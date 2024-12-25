import json

# 读取本地 JSON 文件
input_file = r"D:\files\PHD\myNeRF\nerfstudio\data\12_0\transforms - 副本.json"  # 替换为你的文件名
output_file = r"D:\files\PHD\myNeRF\nerfstudio\data\12_0\mssr.json"  # 替换为输出文件名

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)  # 读取 JSON 数据

# 提取外层参数
camera_params = {
    "camera_angle_x": data["camera_angle_x"],
    "camera_angle_y": data["camera_angle_y"],
    "fl_x": data["fl_x"],
    "fl_y": data["fl_y"],
    "cx": data["cx"],
    "cy": data["cy"],
    "w": data["w"],
    "h": data["h"],
    "num_channels": 25
}

# 将外层参数移入每个 frame
for frame in data["frames"]:
    frame.update(camera_params)

# 删除顶层参数
keys_to_remove = ["camera_angle_x", "camera_angle_y", "fl_x", "fl_y", "cx", "cy", "w", "h"]
for key in keys_to_remove:
    data.pop(key, None)

# 保存修改后的 JSON 文件
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)  # 确保中文等字符不被转义

print(f"处理完成！结果已保存到 {output_file}")