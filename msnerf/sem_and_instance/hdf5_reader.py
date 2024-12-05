import os
import h5py
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import matplotlib.pyplot as plt

class ReadAllHDF5:
    def __init__(self, folder):
        self.folder = folder  # 初始化时传入的文件夹路径
        self.data = {}  # 用于存储每个 h5 文件的内容
        self.database = {}
        self.instance_counts = defaultdict(int)  # 用于记录每个实例的计数
        self.load_hdf5_files()

    def create_database(self):
        sem_ins_path = os.path.join(self.folder, 'YOLO11_Seg', 'hdf5')
        output_path = os.path.join(self.folder, 'YOLO11_Seg', 'database')
        os.makedirs(output_path, exist_ok=True)
        # Gather all .h5 file paths
        file_paths = [
            os.path.join(sem_ins_path, filename)
            for filename in os.listdir(sem_ins_path) if filename.endswith('.h5')
        ]
        # Process files in parallel using ProcessPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.get_instance_data, file_paths)

        for file_prefix, data, instance_count in results:
            self.database[file_prefix] = data
            self.instance_counts[instance_count] += 1  # Track images per instance count

        # Save database as a JSON file
        json_path = os.path.join(output_path, 'database.json')
        with open(json_path, 'w') as json_file:
            json.dump(self.database, json_file, indent=4)

        # Save instance count statistics as JSON file
        counts_path = os.path.join(output_path, 'instance_counts.json')
        with open(counts_path, 'w') as counts_file:
            json.dump(self.instance_counts, counts_file, indent=4)

        print(f"Database saved to {json_path}")
        print(f"Instance count statistics saved to {counts_path}")


    def load_hdf5_files(self):
        # 定义 h5 文件路径
        sem_ins_path = os.path.join(self.folder)### 修改过位置

        # 遍历文件夹下的所有 .h5 文件并加载
        for filename in os.listdir(sem_ins_path):
            if filename.endswith('.h5'):
                file_path = os.path.join(sem_ins_path, filename)
                with h5py.File(file_path, 'r') as h5_file:
                    # 假设每个文件包含 'instance_mask' 数据
                    instance_data = h5_file['instance_mask'][:]
                    # print(instance_data.shape)
                    # 将读取的数据存储在类的属性中
                    file_prefix = os.path.splitext(filename)[0]
                    # print(file_prefix)
                    self.data[file_prefix] = instance_data


    def get_instance_data(self, file_path):
        file_prefix = os.path.splitext(os.path.basename(file_path))[0]
        data = {}
        instances = self.data[file_prefix]
        unique_instances = np.unique(instances)

        # Count valid instances, excluding 0 and 255
        valid_instances = [int(instance_num) for instance_num in unique_instances if
                           instance_num != 0 and instance_num != 255]
        instance_count = len(valid_instances)

        # Initialize each instance's label as None
        for instance_num in valid_instances:
            data[instance_num] = None

        return file_prefix, data, instance_count

    def plot_instance_counts(self):
        # Convert dictionary data to lists for plotting
        save_path = os.path.join(self.folder, 'YOLO11_Seg', 'database','instance_counts_plot.png')
        instance_nums = list(self.instance_counts.keys())
        image_counts = list(self.instance_counts.values())

        # Create bar plot
        plt.bar(instance_nums, image_counts, label="Image Count")

        # Chart title and labels
        plt.title("Instance Counts vs Image Counts")
        plt.xlabel("Instance Count")
        plt.ylabel("Number of Images")
        plt.legend()

        # Save the plot as a PNG file
        plt.savefig(save_path)
        plt.close()  # Close the plot to free memory


if __name__ == '__main__':
    folder_path = r'C:\code\data\001'
    reader = ReadAllHDF5(folder_path)
    reader.create_database()
    reader.plot_instance_counts()