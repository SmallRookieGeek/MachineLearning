import math

import torch.nn as nn
import torch
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, particles):
        # [batch_size, num_particles, embedding_size]
        K = self.query(particles)
        V = self.query(particles)
        Q = self.query(particles)

        # [batch_size, num_particles, num_particles]
        attention_scores = torch.matmul(Q, K.permute(0, 2, 1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # [batch_size, num_particles, num_particles]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = self.dropout(attention_probs)

        # [batch_size, num_particles, embedding_size]
        attention_output = torch.matmul(attention_probs, V)

        return attention_output

class SelfAttentionForProjecion(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttentionForProjecion, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, particles):
        # particles: [batch_size, seq_len, num_particles, embedding_size]
        B, S, N, D = particles.shape  # [B, S, N, D]

        # view as one big batch: [B*S, N, D]
        particles_flat = particles.view(B * S, N, D)

        Q = self.query(particles_flat)  # [B*S, N, D]
        K = self.key(particles_flat)    # [B*S, N, D]
        V = self.value(particles_flat)  # [B*S, N, D]

        # attention: [B*S, N, N]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B*S, N, N]
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        attention_probs = torch.softmax(attention_scores, dim=-1)  # [B*S, N, N]

        attention_output = torch.matmul(attention_probs, V)  # [B*S, N, D]

        # reshape back to [B, S, N, D]
        attention_output = attention_output.view(B, S, N, D)

        return attention_output

class FFN(nn.Module):
    """
    Actually without the FFN layer, there is no non-linearity involved. That is may be why the model cannot fit
    the training queries so well
    """

    def __init__(self, hidden_size, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, particles):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(particles)))))

class ParticleCrusher(nn.Module):

    def __init__(self, embedding_size, num_particles):
        super(ParticleCrusher, self).__init__()

        # self.noise_layer = nn.Linear(embedding_size, embedding_size)
        self.num_particles = num_particles

        self.off_sets = nn.Parameter(torch.zeros([1, 1, num_particles, embedding_size]), requires_grad=True)
        # self.layer_norm = LayerNorm(embedding_size)

    def forward(self, batch_of_embeddings):
        # shape of batch_of_embeddings: [batch_size, embedding_size]
        # the return is a tuple ([batch_size, embedding_size, num_particles], [batch_size, num_particles])
        # The first return is the batch of particles for each entity, the second is the weights of the particles
        # Use gaussian kernel to do this

        batch_size, seq_len, embedding_size = batch_of_embeddings.shape

        # [batch_size, num_particles, embedding_size]
        expanded_batch_of_embeddings = batch_of_embeddings.reshape(batch_size, seq_len, -1, embedding_size) + self.off_sets

        return expanded_batch_of_embeddings
class LogicalTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_len, encoder):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_len)
        #logical
        self.numerical_op_embedding = nn.Embedding(2, d_model)
        self.attribute_embedding = nn.Embedding(1, d_model)
        self.encoder = encoder
        self.numerical_intersection_attn = SelfAttention(d_model)
        self.drop=0.3
        self.dropout = nn.Dropout(self.drop)
        self.numerical_intersection_ffn = FFN(d_model, self.dropout)
        self.numerical_intersection_layer_norm = LayerNorm(d_model)
        #projection
        self.to_particles = ParticleCrusher(d_model, 2)
        self.projection_layer_norm_1 = LayerNorm(d_model)
        self.projection_layer_norm_2 = LayerNorm(d_model)

        self.projection_self_attn = SelfAttentionForProjecion(d_model)

        self.projection_Wz = nn.Linear(d_model, d_model)
        self.projection_Uz = nn.Linear(d_model, d_model)

        self.projection_Wr = nn.Linear(d_model, d_model)
        self.projection_Ur = nn.Linear(d_model, d_model)

        self.projection_Wh = nn.Linear(d_model, d_model)
        self.projection_Uh = nn.Linear(d_model, d_model)

        self.numerical_proj_layer_norm = LayerNorm(d_model)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    def numerical_projection(self, numerical_proj_ids, sub_query_encoding):
        # print("reversed_attribute_projection input shape:", sub_query_encoding.shape)

        # numerical_proj_ids = torch.tensor(numerical_proj_ids)
        numerical_proj_ids = numerical_proj_ids.to(self.numerical_op_embedding.weight.device)

        numerical_embeddings = self.numerical_op_embedding(numerical_proj_ids) #[B,S,D]

        # print("reversed_attribute_projection output shape:", numerical_embeddings.shape)

        sub_query_encoding = self.to_particles(sub_query_encoding)  # [batch_size, seq_len, num_particles, embedding_size]

        Wz = self.projection_Wz
        Uz = self.projection_Uz

        Wr = self.projection_Wr
        Ur = self.projection_Ur

        Wh = self.projection_Wh
        Uh = self.projection_Uh

        relation_transition = torch.unsqueeze(numerical_embeddings, 2)  #[B,S,1,D]

        projected_particles = sub_query_encoding

        z = self.sigmoid(Wz(self.dropout(relation_transition)) + Uz(self.dropout(projected_particles)))
        r = self.sigmoid(Wr(self.dropout(relation_transition)) + Ur(self.dropout(projected_particles)))

        h_hat = self.tanh(Wh(self.dropout(relation_transition)) + Uh(self.dropout(projected_particles * r)))

        h = (1 - z) * projected_particles + z * h_hat

        projected_particles = h
        projected_particles = self.projection_layer_norm_1(projected_particles) #[B,S,2,D]

        projected_particles = self.projection_self_attn(self.dropout(projected_particles)).sum(dim=2) #[B,S,D]

        projected_particles = self.numerical_proj_layer_norm(projected_particles)

        return projected_particles

    def intersection(self, emb1, emb2):
        # Intersection of values
        all_subquery_encodings = torch.cat([emb1,emb2], dim=1)
        # print(all_subquery_encodings.shape)
        batch_size, num_sets, embedding_size = all_subquery_encodings.shape

        flatten_particles = all_subquery_encodings.view(batch_size, -1, embedding_size) #[B,S,D]

        flatten_particles = self.numerical_intersection_attn(self.dropout(flatten_particles))
        flatten_particles = self.numerical_intersection_layer_norm(flatten_particles)
        flatten_particles = self.numerical_intersection_ffn(flatten_particles) + flatten_particles
        flatten_particles = self.numerical_intersection_layer_norm(flatten_particles)

        encoding = flatten_particles.sum(dim=1)

        return encoding

    def forward(self, x, p):
        x = self.input_proj(x)  # (B, S, D)
        x = x.permute(1, 0, 2)  # (S, B, D)
        out = self.transformer(x)

        #logic
        p_emb = self.encoder(p) #[B,S,D]
        batch_size, seq_len = p_emb.size(0),p_emb.size(1)
        type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long).to(device)
        small_ids = torch.zeros((batch_size, seq_len), dtype=torch.long).to(device)
        big_ids = torch.ones((batch_size,seq_len), dtype=torch.long).to(device)

        type_embs = self.attribute_embedding(type_ids) #[B,S,D]
        power_emb = p_emb + type_embs
        proj1 = self.numerical_projection(small_ids,power_emb)
        proj2 = self.numerical_projection(big_ids,power_emb)

        out2 = self.intersection(proj1,proj2)

        final_out = out[-1] + out2

        out = self.fc(final_out)  # 使用最后一个时间步

        return out