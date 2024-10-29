# src/data_preprocessing.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
import pandas as pd
import tqdm as tqdm

class TrafficDataset(Dataset):
    """自定义网络流量数据集类，适应输入形状为 [用户数, 时间步长, 网络指标]"""

    def __init__(self, data, sequence_length=30):
        # 假设数据的形状为 [用户数, 时间步长, 网络指标]
        self.sequence_length = sequence_length
        self.scalers = [MinMaxScaler() for _ in range(data.shape[2])]  # 为每个网络指标创建一个Scaler

        # 对每个网络指标归一化
        data_normalized = []
        for i in range(data.shape[2]):
            normalized_column = self.scalers[i].fit_transform(data[:, :, i])
            data_normalized.append(normalized_column)

        # 转置回原始形状
        self.data_normalized = np.stack(data_normalized, axis=-1)

    def __len__(self):
        # 返回数据长度
        return self.data_normalized.shape[1] - self.sequence_length

    def __getitem__(self, index):
        # 获取一个时间序列片段和对应的目标值
        sequence = self.data_normalized[:, index:index + self.sequence_length, :]
        target = self.data_normalized[:, index + self.sequence_length, :]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def inverse_transform(self, data, metric_idx):
        # 数据反归一化
        return self.scalers[metric_idx].inverse_transform(data)


def txt_to_npy_optimized(txt_path, npy_path, num_users=142, num_services=4500, num_time_slices=64):
    """
    高效地将 tpdata.txt 文件中的数据转换为 .npy 格式的三维数组。

    参数:
        txt_path (str): 输入 .txt 文件的路径。
        npy_path (str): 输出 .npy 文件的路径。
        num_users (int): 用户数量，默认为 142。
        num_services (int): 服务数量，默认为 4500。
        num_time_slices (int): 时间片数量，默认为 64。

    返回:
        None
    """
    # 读取 txt 文件，将其解析为 DataFrame
    data = pd.read_csv(txt_path, sep=r'\s+', header=None,
                       names=['User ID', 'Service ID', 'Time Slice ID', 'Throughput'])

    # 减去1，转化为零索引，以匹配数组索引
    data['User ID'] -= 1
    data['Service ID'] -= 1
    data['Time Slice ID'] -= 1

    # 创建一个三维数组，大小为 [用户数, 时间步长, 服务数]
    throughput_data = np.zeros((num_users, num_time_slices, num_services), dtype=np.float32)

    # 使用 DataFrame 的 `pivot_table` 将数据重塑为三维结构
    pivot_table = data.pivot_table(index='User ID', columns=['Time Slice ID', 'Service ID'], values='Throughput',
                                   fill_value=0)

    # 将 `pivot_table` 转换为 numpy 数组，并重新调整形状
    reshaped_data = pivot_table.values.reshape(num_users, num_time_slices, num_services)

    # 将重塑的数据赋值到 throughput_data 数组
    throughput_data[:reshaped_data.shape[0], :reshaped_data.shape[1], :reshaped_data.shape[2]] = reshaped_data

    # 保存为 .npy 文件
    np.save(npy_path, throughput_data)
    print(f"数据已成功保存为 {npy_path}")


def txt_to_npy(txt_path, npy_path, num_users=142, num_services=4500, num_time_slices=64):
    """
    将 tpdata.txt 文件中的数据转换为 .npy 格式的三维数组。

    参数:
        txt_path (str): 输入 .txt 文件的路径。
        npy_path (str): 输出 .npy 文件的路径。
        num_users (int): 用户数量，默认为 142。
        num_services (int): 服务数量，默认为 4500。
        num_time_slices (int): 时间片数量，默认为 64。

    返回:
        None
    """
    # 读取 txt 文件，将其解析为 DataFrame
    data = pd.read_csv(txt_path, sep=r'\s+', header=None,
                       names=['User ID', 'Service ID', 'Time Slice ID', 'Throughput'], encoding='utf-8')

    # 初始化一个三维数组，大小为 [用户数, 时间步长, 服务数]
    throughput_data = np.zeros((num_users, num_time_slices, num_services), dtype=np.float32)

    # 将数据填充到三维数组中
    for _, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        user_id = int(row['User ID']) - 1  # 假设用户ID从1开始
        service_id = int(row['Service ID']) - 1  # 假设服务ID从1开始
        time_slice_id = int(row['Time Slice ID']) - 1  # 假设时间片ID从1开始
        throughput = float(row['Throughput'])

        # 填充 throughput 数据到三维数组的对应位置
        throughput_data[user_id, time_slice_id, service_id] = throughput

    # 保存为 .npy 文件
    np.save(npy_path, throughput_data)
    print(f"数据已成功保存为 {npy_path}")


# 加载数据
def load_data(file_path):
    data = np.load(file_path)
    return data

if __name__ == '__main__':
    # 将 tpdata.txt 转换为 .npy 文件
    txt_to_npy(txt_path='../data/raw_data/tpdata.txt', npy_path='../data/processed_data/tpdata.npy')
    # 加载数据
    data = load_data('../data/processed_data/tpdata.npy')
    print(f"数据形状: {data.shape}")
    print(f"数据示例: {data[0, 0, :]}")
    dataset = TrafficDataset(data, sequence_length=30)
    print(f"数据集长度: {len(dataset)}")
    sequences, targets = dataset[0]
    print(f"时间序列形状: {sequences.shape}")
    print(f"目标值形状: {targets.shape}")
    print(f"反归一化示例: {dataset.inverse_transform(targets, metric_idx=0)}")
