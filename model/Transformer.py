import torch.nn as nn
class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_len):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_len)
    def forward(self, x):
        x = self.input_proj(x)  # (B, T, D)
        x = x.permute(1, 0, 2)  # (T, B, D)
        out = self.transformer(x)
        out = self.fc(out[-1])  # 使用最后一个时间步
        return out