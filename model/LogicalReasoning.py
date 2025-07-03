import torch.nn as nn
import torch
import torch.nn.functional as F
class GammaIntersection(nn.Module):

    def __init__(self, dim):
        super(GammaIntersection, self).__init__()
        self.dim = dim
        self.layer_alpha1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_beta1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_alpha2 = nn.Linear(self.dim, self.dim)
        self.layer_beta2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_alpha1.weight)
        nn.init.xavier_uniform_(self.layer_beta1.weight)
        nn.init.xavier_uniform_(self.layer_alpha2.weight)
        nn.init.xavier_uniform_(self.layer_beta2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)#(num_item, batch_size, dim*2)
        layer1_alpha = F.relu(self.layer_alpha1(all_embeddings))  # (num_item, batch_size, dim * 2)
        attention1 = F.softmax(self.layer_alpha2(layer1_alpha), dim=0)  # (num_item, batch_size, dim)

        layer1_beta = F.relu(self.layer_beta1(all_embeddings))  # (num_item, batch_size, dim * 2)
        attention2 = F.softmax(self.layer_beta2(layer1_beta), dim=0)  # (num_item, batch_size, dim)

        alpha_embedding = torch.sum(attention1 * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention2 * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding
class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)
class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
class TransformerWithLogic(nn.Module):
    def __init__(self, input_size, emb_size, nhead, num_layers, output_len):
        super().__init__()
        self.input_proj = nn.Linear(input_size, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model = emb_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_size, output_len)
        #gamma embedding
        self.logic_linear = nn.Linear(input_size, emb_size)
        self.logic_linear2 = nn.Linear(1, emb_size)
        self.intersection = GammaIntersection(emb_size)
        self.fea2log = nn.Linear(emb_size, 2 * emb_size, bias=False)
        self.gamma_regularizer = Regularizer(1, 0.15, 1e9)
        self.fc2 = nn.Linear(emb_size, output_len)
        self.loss_fct = BPRLoss()
    def feature_to_gamma_64(self, feature):

        feature = self.fea2log(feature)
        emb = self.gamma_regularizer(feature)
        alpha, beta = torch.chunk(emb, 2, dim=-1)

        return alpha, beta
    def gamma_vec_to_dis(self, alpha, beta):

        return torch.distributions.gamma.Gamma(alpha, beta)

        return dis
    def forward(self, input):
        # feature
        x = self.input_proj(input)  # (B, T, D)
        x = x.permute(1, 0, 2)  # (T, B, D)
        out = self.transformer(x)
        out = self.fc(out[-1])  # 使用最后一个时间步

        # logical
        feature = self.logic_linear(input)
        alpha_seq, beta_seq = self.feature_to_gamma_64(feature)
        alpha_embedding_list = []
        beta_embedding_list = []
        for i in range(0, alpha_seq.size(1)):
            alpha_embedding_list.append(alpha_seq[:, i])
            beta_embedding_list.append(beta_seq[:, i])
        alpha_output, beta_output = self.intersection(torch.stack(alpha_embedding_list),
                                                             torch.stack(beta_embedding_list))
        logic_output = alpha_output / beta_output
        # out_dis = self.gamma_vec_to_dis(alpha_output, beta_output)  # [bs, es]
        logic_out = self.fc2(logic_output)

        out = out + logic_out

        return out