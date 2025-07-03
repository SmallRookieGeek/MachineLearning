import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len)
    def forward(self, x):
        # output, (h_n, c_n) = lstm(x)
        # output 是每个时间步的输出 [batch_size, seq_len, hidden_dim]
        # h_n 是每层最后一个时间步的隐藏状态  [batch_size, hidden_dim]
        # c_n 是最后一个时间步的记忆单元状态
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1]) #[batch_size,90]
        return out