# src/predict.py

import torch
from torch.utils.data import DataLoader
from data_preprocessing import TrafficDataset, load_data
from model import CRNN


def predict(data_path, sequence_length=30, model_path='../results/checkpoints/crnn_model.pth'):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据和模型
    data = load_data(data_path)
    dataset = TrafficDataset(data, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_size = data.shape[2]  # 网络指标数
    model = CRNN(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for sequences, _ in dataloader:
            sequences = sequences.to(device)
            output = model(sequences)
            predictions.append(output.squeeze(0).cpu().numpy())  # 将结果转移回CPU并转换为numpy

    # 反归一化和结果处理保持不变
    predictions = np.array(predictions)
    inversed_predictions = []
    for i in range(input_size):
        inversed_predictions.append(dataset.inverse_transform(predictions[:, :, i], metric_idx=i))
    inversed_predictions = np.stack(inversed_predictions, axis=-1)

    return inversed_predictions
