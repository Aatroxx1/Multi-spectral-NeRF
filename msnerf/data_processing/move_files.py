import threading

import pandas as pd
import os
import shutil

def act(frames, down, new):
    os.makedirs(os.path.join(new, "images"),exist_ok=True)
    os.makedirs(os.path.join(new, "images_for_sfm"), exist_ok=True)
    # shutil.copytree(frames, os.path.join(new, "images"), dirs_exist_ok=True)
    shutil.copytree(down, os.path.join(new, "images_for_sfm"), dirs_exist_ok=True)
    shutil.copytree(down, os.path.join(new, "images"), dirs_exist_ok=True)

if __name__ == '__main__':
    cvs_path = r"/data/编号.csv"
    input_dir = r"E:\SeMs-NeRF"
    output_dir = r"/data"
    df = pd.read_csv(cvs_path)
    # 遍历每一行
    threads = []
    for index, row in df.iterrows():
        input_frames = os.path.join(input_dir, row['old']+ "_frames")
        input_down = os.path.join(input_dir, row['old'] + "_downsampled_frames")
        output_group_dir = row['new']
        output_group_path = os.path.join(output_dir, output_group_dir)
        t = threading.Thread(target=act, args=(input_frames, input_down,output_group_path,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("All tasks completed.")