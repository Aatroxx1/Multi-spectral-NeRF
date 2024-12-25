import cv2
import os


def extract_frames(video_path, output_folder, frame_rate=1):
    """
    将视频拆分成照片并保存到指定文件夹。

    参数:
    - video_path: 输入视频的路径
    - output_folder: 输出照片的文件夹路径
    - frame_rate: 每隔多少帧保存一张图片，默认值为1（每帧保存）。
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"视频文件 {video_path} 不存在！")
        return

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频 {video_path}")
        return

    # 获取视频的帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {fps:.2f} FPS，总帧数: {total_frames}")

    # 初始化帧计数器
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取结束

        # 每隔 `frame_rate` 帧保存一次图片
        if frame_count % frame_rate == 0:
            # 保存帧为图片
            output_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"保存帧 {frame_count} 为图片: {output_path}")

        frame_count += 1

    # 释放视频对象
    cap.release()
    print(f"完成！共保存了 {saved_count} 张图片到文件夹: {output_folder}")


# 示例用法
if __name__ == "__main__":
    video_path = r"D:\files\PHD\myNeRF\nerfstudio\data\VID20241220203642\VID20241220203642.mp4"  # 替换为你的视频文件路径
    output_folder = r"D:\files\PHD\myNeRF\nerfstudio\data\VID20241220203642\images"   # 替换为输出图片的文件夹
    frame_rate = 2 # 每隔30帧保存一张图片（帧率可根据需要调整）

    extract_frames(video_path, output_folder, frame_rate)