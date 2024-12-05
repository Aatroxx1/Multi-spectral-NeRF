import os
import cv2
import numpy as np
from scipy.ndimage import median_filter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 用于显示进度条

# 1. 指定主文件夹路径
main_folder_path = r"E:\SeMs-NeRF\20240126_rapeseed_MUtiPoint"  # 替换为包含子文件夹的主文件夹路径

# 2. 获取所有子文件夹中的视频文件列表
all_files = []
for root, dirs, files in os.walk(main_folder_path):
    for file in files:
        if file.endswith(".avi"):
            all_files.append(os.path.join(root, file))

# 检查是否有视频文件
if not all_files:
    raise FileNotFoundError("指定路径下没有找到任何AVI文件")

# 总视频文件数量
num_files = len(all_files)

# 用户指定每X秒提取一帧的时间间隔
X = float(input("请输入时间间隔X（秒）："))  # 用户输入时间间隔，单位为秒

# 定义处理视频的函数
def process_video(video_file):
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # 读取视频
    vid_obj = cv2.VideoCapture(video_file)
    frame_rate = vid_obj.get(cv2.CAP_PROP_FPS)
    num_frames = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)

    # 计算要跳过的帧数
    time_step_frames = round(frame_rate * X)

    # 创建保存帧的文件夹
    output_folder = os.path.join(os.path.dirname(video_file), f"{video_name}")
    os.makedirs(output_folder, exist_ok=True)

    # 创建保存降采样图像的文件夹
    downsampled_folder = os.path.join(output_folder, "images_for_sfm")
    os.makedirs(downsampled_folder, exist_ok=True)

    frames_folder = os.path.join(output_folder, "images")

    # 创建保存拆分和上采样部分的文件夹
    num_parts = 25  # 分成25个部分
    part_folder_prefix = "part"  # 文件夹前缀
    for i in range(1, num_parts + 1):
        part_folder = os.path.join(frames_folder, f"{part_folder_prefix}{i}")
        os.makedirs(part_folder, exist_ok=True)

    # 循环提取每X秒的帧并进行处理
    frame_idx = 1
    bands_order = np.array([
        [21, 22, 23, 24, 25],
        [16, 17, 18, 19, 20],
        [11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5]
    ])

    while True:
        ret, frame = vid_obj.read()

        if not ret:
            break

        # 删除最后三行和最后三列
        frame = frame[:-3, :-3, :]

        # 对帧进行中值滤波
        filtered_frame = median_filter(frame, size=(5, 5, 1))

        # 对中值滤波后的帧进行降采样
        downsampled_frame = cv2.resize(filtered_frame, (0, 0), fx=0.2, fy=0.2)

        # 保存降采样后的图像
        downsampled_output_filename = os.path.join(
            downsampled_folder, f"frame_{frame_idx:04d}.png"
        )
        cv2.imwrite(downsampled_output_filename, downsampled_frame)

        # 拆分图像
        DN_white_in_bands = np.zeros(
            (frame.shape[0] // 5, frame.shape[1] // 5, num_parts), dtype=np.uint8
        )

        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                r = row % 5
                c = col % 5
                band_index = bands_order[r, c] - 1
                DN_white_in_bands[row // 5, col // 5, band_index] = frame[row, col, 0]

        # 保存图像部分和上采样部分
        for i in range(num_parts):
            current_part = DN_white_in_bands[:, :, i]

            # 保存原始部分
            part_output_filename = os.path.join(
                frames_folder, f"{part_folder_prefix}{i+1}", f"frame_{frame_idx:04d}.png"
            )
            cv2.imwrite(part_output_filename, current_part)

        # 跳过指定的帧数
        for _ in range(time_step_frames - 1):
            if not vid_obj.read()[0]:
                break

        frame_idx += 1

    vid_obj.release()
    print(f"视频 {video_name} 的帧提取和图像处理完成！")

# # 使用多线程并行处理所有视频
# with ThreadPoolExecutor() as executor:
#     list(tqdm(executor.map(process_video, all_files), total=num_files))

process_video(r"E:\SeMs-NeRF\20240126_rapeseed_MUtiPoint\2024qiu1.avi")

print("所有视频的处理完成！")