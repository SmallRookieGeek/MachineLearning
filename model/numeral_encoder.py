import torch
import numpy as np
import networkx as nx
from torch import nn
from torch.nn import CrossEntropyLoss
class NumeralEncoder(torch.nn.Module):
    """
    This class is the base class for all numeral encoders.
    """

    def __init__(self, output_size, all_values):
        super(NumeralEncoder, self).__init__()

        self.output_size = output_size

        self.all_values = all_values

        all_values_f32 = [float(np.float32(v)) for v in all_values]
        value_vocab = {v: i for i, v in enumerate(all_values_f32)}
        # value_vocab = dict(zip(all_values, range(0, len(all_values))))
        self.value_vocab = value_vocab

    def get_embeddings(self):
        """
        This function returns the embeddings of the test set.
        """
        # if torch.cuda.is_available() :
        #     return self.forward( torch.tensor(self.all_values).cuda() )
        # else:
        #     return self.forward( torch.tensor(self.all_values) )

        return self.forward(self.all_values)

    def values2ids(self, values: list):
        """
        This function converts the values of the numeral encoder to ids.
        """
        ids = []

        for value in values:
            # print("values: " + str(value))
            # print("value2id: " + str(self.value_vocab))
            ids.append(self.value_vocab[value])
        return ids

    def value2id(self, value):
        """
        This function converts a value to an id.
        """
        return self.value_vocab[value]

    def forward(self, x):
        """
        This function is the forward pass of the numeral encoder.
        """
        raise NotImplementedError

class PositionalEncoder(NumeralEncoder):
    """
    This class is the positional encoder.
    """

    def __init__(self, output_size, all_values, n=10000, scaler="log"):
        super(PositionalEncoder, self).__init__(output_size, all_values)
        self.n = n

        self.scaler_name = scaler

        if scaler == "log":
            self.scaler = log_scaling_function
        elif scaler == "quantile":
            self.scaler = QuantileScaling(self.train_values).quantile_scaling_function

        if torch.cuda.is_available():
            self.sinosoidal_embedding = torch.nn.Embedding(len(self.all_values), output_size).cuda()
        else:
            self.sinosoidal_embedding = torch.nn.Embedding(len(self.all_values), output_size)

        all_values_tensor = torch.tensor(self.all_values)
        if torch.cuda.is_available():
            all_values_tensor = all_values_tensor.cuda()

        d = self.output_size

        x = self.scaler(all_values_tensor)  #用对数压缩一下大小
        denominator = 1 / torch.pow(self.n, 2 * torch.arange(0, d // 2).float().to(x.device) / d).unsqueeze(0)
        x = x.unsqueeze(1)

        # print("x: ", x.shape)
        # print("denominator: ", denominator.shape)
        # print(self.sinosoidal_embedding.weight.shape)
        self.sinosoidal_embedding.weight.requires_grad = False
        self.sinosoidal_embedding.weight[:, 0::2] = torch.sin(x * denominator)  # [all_nv,D]
        self.sinosoidal_embedding.weight[:, 1::2] = torch.cos(x * denominator)

    # def forward(self, x):
    #     """
    #     This function is the forward pass of the positional encoder.
    #
    #     x: the input array of numbers to be encoded
    #     """
    #
    #     if isinstance(x, list):
    #         x_ids = self.values2ids(x)
    #     else:
    #         x_ids = self.values2ids([x])
    #
    #     if torch.cuda.is_available():
    #         x_ids = torch.tensor(x_ids).cuda()
    #     else:
    #         x_ids = torch.tensor(x_ids)
    #
    #     return self.sinosoidal_embedding(x_ids)
    def batch_values2ids(self, x: torch.Tensor):
        """
        Converts a 2D tensor of values (batch_size, seq_len) to ids using value_vocab.
        """
        x_np = x.cpu().numpy()  # shape: (batch_size, seq_len)

        id_matrix = []
        for row in x_np:
            row_ids = self.values2ids(list(row))
            id_matrix.append(row_ids)

        return torch.tensor(id_matrix, dtype=torch.long, device=x.device)
    def forward(self, x):
        """
           x: torch.Tensor of shape (batch_size, seq_len), containing raw float/int values
           """
        assert isinstance(x, torch.Tensor), "Input x must be a torch tensor"
        x_ids = self.batch_values2ids(x)
        return self.sinosoidal_embedding(x_ids)
class QuantileScaling():
    def __init__(self, train_values):
        self.train_values_sorted = np.sort(train_values, axis=None)
        self.train_values_sorted = torch.from_numpy(self.train_values_sorted).float()

    def quantile_scaling_function(self, x):
        self.train_values_sorted = self.train_values_sorted.to(x.device)

        # print("x: ", x)
        # print("train_values_sorted: ", self.train_values_sorted)

        num_smaller = torch.sum((x.reshape(-1, 1) - self.train_values_sorted.reshape(1, -1) > 0), dim=1)

        # print("num_smaller: ", num_smaller)

        largest_smaller_index = num_smaller - 1
        # print("largest_smaller_index: ", largest_smaller_index)

        largest_smaller_index[largest_smaller_index < 0] = 0
        # print("value shape: ", self.train_values_sorted.shape[0])
        # largest_smaller_index[largest_smaller_index > 0] = 0
        largest_smaller_index[largest_smaller_index >= self.train_values_sorted.shape[0] - 2] = \
        self.train_values_sorted.shape[0] - 2

        next_index = largest_smaller_index + 1

        largest_smaller_index_value = self.train_values_sorted[largest_smaller_index]
        next_index_value = self.train_values_sorted[next_index]

        approximate_indices = largest_smaller_index + (x - largest_smaller_index_value) / (
                    next_index_value - largest_smaller_index_value)

        percentile = approximate_indices / len(self.train_values_sorted)

        percentile[percentile < 0] = 0
        percentile[percentile > 1] = 1

        return percentile * 64


def log_scaling_function(x):  #区间 [-1, 1] 内的值保持不变；区间 (1, +∞) 内的值被对数压缩成平滑曲线；区间 (-∞, -1) 内的值也被对称地压缩 成平滑曲线。

    # print("x: " + str(x))
    x[x > 1] = torch.log(x[x > 1]) + 1
    x[x < -1] = - torch.log(-x[x < -1]) - 1
    # print("x: " + str(x))
    return x