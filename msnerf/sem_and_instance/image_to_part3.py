import cv2
import os
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time


class ImageProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.channel_dirs = None
        self.processing_lock = Lock()  # 用于确保线程安全
        self.processed_count = 0
        self.total_files = 0

    def create_channel_dirs(self):
        """创建用于存储不同通道图片的子目录"""
        # 创建三个子目录
        part1_dir = os.path.join(self.output_dir, "part1")  # R通道
        part2_dir = os.path.join(self.output_dir, "part2")  # G通道
        part3_dir = os.path.join(self.output_dir, "part3")  # B通道

        # 确保所有目录都存在
        os.makedirs(part1_dir, exist_ok=True)
        os.makedirs(part2_dir, exist_ok=True)
        os.makedirs(part3_dir, exist_ok=True)

        self.channel_dirs = {
            'part1': part1_dir,
            'part2': part2_dir,
            'part3': part3_dir
        }

    def process_single_image(self, input_path):
        """
        处理单张图片，将其分离为R、G、B通道并分别保存为灰度图到对应子目录

        参数:
            input_path: 输入图片的路径
        """
        try:
            # 使用NumPy和OpenCV读取图片
            img = cv2.imread(input_path)
            if img is None:
                print(f"无法读取图片: {input_path}")
                return False

            # 使用NumPy进行通道分离（更高效）
            b = img[:, :, 0]
            g = img[:, :, 1]
            r = img[:, :, 2]

            # 获取原始文件名（不含扩展名）
            base_name = Path(input_path).stem

            # 使用线程锁确保文件保存的原子性
            with self.processing_lock:
                # 保存各个通道到对应的子目录，保持原有的命名方式
                cv2.imwrite(os.path.join(self.channel_dirs['part1'], f"{base_name}_part1.png"), r)
                cv2.imwrite(os.path.join(self.channel_dirs['part2'], f"{base_name}_part2.png"), g)
                cv2.imwrite(os.path.join(self.channel_dirs['part3'], f"{base_name}_part3.png"), b)

                # 更新处理计数
                self.processed_count += 1
                # 打印进度
                print(f"进度: {self.processed_count}/{self.total_files} - 成功处理图片: {input_path}")
                print(f"- 已保存R通道到: {os.path.join(self.channel_dirs['part1'], f'{base_name}_part1.png')}")
                print(f"- 已保存G通道到: {os.path.join(self.channel_dirs['part2'], f'{base_name}_part2.png')}")
                print(f"- 已保存B通道到: {os.path.join(self.channel_dirs['part3'], f'{base_name}_part3.png')}")

            return True

        except Exception as e:
            print(f"处理图片时发生错误 {input_path}: {str(e)}")
            return False

    def process_all_images(self):
        """处理所有图片"""
        # 创建输出目录结构
        os.makedirs(self.output_dir, exist_ok=True)
        self.create_channel_dirs()

        # 支持的图片格式（大小写都支持）
        image_patterns = [
            "*.png", "*.PNG",
            "*.jpg", "*.JPG",
            "*.jpeg", "*.JPEG",
            "*.bmp", "*.BMP",
            "*.tiff", "*.TIFF"
        ]

        # 获取所有图片文件
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(self.input_dir, pattern)))

        # 去重（防止大小写重复）
        image_files = list(set(image_files))

        if not image_files:
            print(f"在 {self.input_dir} 目录下没有找到任何图片文件")
            return

        self.total_files = len(image_files)
        success_count = 0

        # 记录开始时间
        start_time = time.time()

        # 使用线程池处理图片
        # 线程数设置为处理器核心数的2倍（可以根据需要调整）
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            # 提交所有任务并等待完成
            results = list(executor.map(self.process_single_image, image_files))

        # 统计成功处理的数量
        success_count = sum(1 for result in results if result)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 输出处理结果统计
        print(f"\n处理完成！")
        print(f"总共处理: {self.total_files} 张图片")
        print(f"成功处理: {success_count} 张图片")
        print(f"失败处理: {self.total_files - success_count} 张图片")
        print(f"总处理时间: {processing_time:.2f} 秒")
        print(f"平均每张图片处理时间: {processing_time / self.total_files:.3f} 秒")
        print(f"\n处理后的图片保存在以下目录:")
        print(f"R通道图片: {self.channel_dirs['part1']}")
        print(f"G通道图片: {self.channel_dirs['part2']}")
        print(f"B通道图片: {self.channel_dirs['part3']}")


def image_to_part3(input_directory):
    """
    将指定目录下的图片处理为三个通道并保存到相应目录

    参数:
        input_directory: 输入图片所在目录的路径
    """
    # 自动生成输出目录路径（在输入目录同级创建images目录）
    output_directory = os.path.join(os.path.dirname(input_directory), "images")

    # 创建处理器实例并执行处理
    processor = ImageProcessor(input_directory, output_directory)
    processor.process_all_images()


if __name__ == "__main__":
    # 设置输入文件夹路径
    input_folder = r"D:\files\PHD\myNeRF\nerfstudio\data\Semantic\raw"
    # 处理图片
    image_to_part3(input_folder)