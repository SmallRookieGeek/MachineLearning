import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch
def load_and_process(file_path):

    df = pd.read_csv(file_path, parse_dates=['DateTime'])

    # 将所有列名统一转为小写，防止大小写不一致报错
    df.columns = [col.lower() for col in df.columns]

    # 强制将数值列转换为 float（避免某列是字符串）
    numeric_cols = [
        'global_active_power', 'global_reactive_power',
        'voltage', 'global_intensity',
        'sub_metering_1', 'sub_metering_2', 'sub_metering_3',
        'rr', 'nbjrr1', 'nbjrr5', 'nbjrr10', 'nbjbrou'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 日期聚合
    df['date'] = df['datetime'].dt.date
    daily_df = df.groupby('date').agg({
        'global_active_power': 'sum',
        'global_reactive_power': 'sum',
        'sub_metering_1': 'sum',
        'sub_metering_2': 'sum',
        'sub_metering_3': 'sum',
        'voltage': 'mean',
        'global_intensity': 'mean',
        'rr': 'first',
        'nbjrr1': 'first',
        'nbjrr5': 'first',
        'nbjrr10': 'first',
        'nbjbrou': 'first'
    }).reset_index()

    # 计算剩余功率
    daily_df['sub_metering_remainder'] = (
        daily_df['global_active_power'] * 1000 / 60 -
        (daily_df['sub_metering_1'] + daily_df['sub_metering_2'] + daily_df['sub_metering_3'])
    )

    return daily_df

# def load_and_process_test(file_path):
#
#     column_names = [
#         'DateTime',
#         'Global_active_power',
#         'Global_reactive_power',
#         'Voltage',
#         'Global_intensity',
#         'Sub_metering_1',
#         'Sub_metering_2',
#         'Sub_metering_3',
#         'RR',
#         'NBJRR1',
#         'NBJRR5',
#         'NBJRR10',
#         'NBJBROU'
#     ]
#     df = pd.read_csv(file_path, header=None, names=column_names, parse_dates=['DateTime'])
#
#     # 将所有列名统一转为小写，防止大小写不一致报错
#     df.columns = [col.lower() for col in df.columns]
#
#     # 强制将数值列转换为 float（避免某列是字符串）
#     numeric_cols = [
#         'global_active_power', 'global_reactive_power',
#         'voltage', 'global_intensity',
#         'sub_metering_1', 'sub_metering_2', 'sub_metering_3',
#         'rr', 'nbjrr1', 'nbjrr5', 'nbjrr10', 'nbjbrou'
#     ]
#     df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
#
#     # 日期聚合
#     df['date'] = df['datetime'].dt.date
#     daily_df = df.groupby('date').agg({
#         'global_active_power': 'sum',
#         'global_reactive_power': 'sum',
#         'sub_metering_1': 'sum',
#         'sub_metering_2': 'sum',
#         'sub_metering_3': 'sum',
#         'voltage': 'mean',
#         'global_intensity': 'mean',
#         'rr': 'first',
#         'nbjrr1': 'first',
#         'nbjrr5': 'first',
#         'nbjrr10': 'first',
#         'nbjbrou': 'first'
#     }).reset_index()
#
#     # 计算剩余功率
#     daily_df['sub_metering_remainder'] = (
#             daily_df['global_active_power'] * 1000 / 60 -
#             (daily_df['sub_metering_1'] + daily_df['sub_metering_2'] + daily_df['sub_metering_3'])
#     )
#
#     return daily_df
def create_sequences(df, input_len=90, output_len=90):
    # features = df.drop(columns=['date', 'global_active_power']).values  #values 将 DataFrame 转换成 numpy.ndarray 此处是二维数组，[总天数，特征数]
    features = df.drop(columns=['date']).values
    targets = df['global_active_power'].values
    X, Y = [], [] #存特征和labels,形状为[len(df)-input_len-output_len,90,feature_num]
    for i in range(len(df) - input_len - output_len + 1):
        X.append(features[i:i+input_len])
        Y.append(targets[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(Y)

def create_logic_sequences(df, input_len=90, output_len=90):
    features = df.drop(columns=['date', 'global_active_power']).values  #values 将 DataFrame 转换成 numpy.ndarray 此处是二维数组，[总天数，特征数]s
    targets = df['global_active_power'].values
    X, Y = [], [] #存特征和labels,形状为[len(df)-input_len-output_len,90,feature_num]
    for i in range(len(df) - input_len - output_len + 1):
        X.append(features[i:i+input_len])
        Y.append(targets[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(Y)
def create_power_sequences(df, input_len=90, output_len=90):

    powers = df['global_active_power'].values
    power = []
    for i in range(len(df) - input_len - output_len + 1):
        power.append(powers[i:i+input_len])
    return np.array(power)

def split_train_val(X_train, Y_train, batch_size, shuffle, val_ratio=0.1):
    total_len = len(X_train)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    X_train_split = X_train[:train_len]
    Y_train_split = Y_train[:train_len]
    X_val = X_train[train_len:]
    Y_val = Y_train[train_len:]

    train_loader = DataLoader(TensorDataset(X_train_split, Y_train_split), batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

    return train_loader, val_loader

def split_train_val_logic(X_train, Y_train, P_train, batch_size, shuffle, val_ratio=0.1):
    total_len = len(X_train)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    X_train_split = X_train[:train_len]
    Y_train_split = Y_train[:train_len]
    X_val = X_train[train_len:]
    Y_val = Y_train[train_len:]

    P_train_split = P_train[:train_len]
    P_val = P_train[train_len:]

    train_loader = DataLoader(TensorDataset(X_train_split, Y_train_split, P_train_split), batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(TensorDataset(X_val, Y_val, P_val), batch_size=batch_size)

    return train_loader, val_loader

def get_all_values(df,df2):

    unique_prices = df['global_active_power'].unique()
    unique_prices = unique_prices.tolist()

    unique_prices2 = df2['global_active_power'].unique()
    unique_prices2 = unique_prices2.tolist()

    combined_prices = unique_prices + unique_prices2
    unique_combined_prices = sorted(set(combined_prices))
    return unique_combined_prices
class Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32) #[train_num,feature]
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LogicalDataset(Dataset):
    def __init__(self, X, Y, power):
        self.X = torch.tensor(X, dtype=torch.float32)  # [train_num,feature]
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.P = torch.tensor(power, dtype=torch.float32)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.P[idx]