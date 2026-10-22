import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingTransformer(nn.Module):
    def __init__(self, num_agents, d_model, n_layer, n_head, dim_ffm):
        super().__init__()
        self.n = num_agents
        self.d_model = d_model
        self.n_layer = n_layer

        self.in_proj = nn.Linear(2, d_model)

        self.row_emb = nn.Embedding(self.n, d_model)
        self.col_emb = nn.Embedding(self.n, d_model)

        self.rows = nn.ModuleList()
        self.cols = nn.ModuleList()
        self.fuse = nn.ModuleList()

        for _ in range(n_layer):
            row_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=dim_ffm, 
                dropout=0.0, activation="relu", batch_first=True, norm_first=True
            )

            col_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=dim_ffm, 
                dropout=0.0, activation="relu", batch_first=True, norm_first=True
            )

            self.rows.append(row_layer)
            self.cols.append(col_layer)
            self.fuse.append(nn.Sequential(
                nn.Linear(2*d_model, dim_ffm),
                nn.ReLU(),
                nn.Linear(dim_ffm, d_model)
            ))
        
        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, P, Q):
        B, n, _ = P.shape
        x = torch.stack([P, Q], dim=-1)
        x = self.in_proj(x)
        
        rows = torch.arange(n, device=P.device)
        cols = torch.arange(n, device=P.device)
        x = x + self.row_emb(rows)[None, :, None, :] + self.col_emb(cols)[None, None, :, :]

        valid = (P > 0) & (Q > 0)

        for layer in range(self.n_layer):
            d = x.shape[-1]

            row_x = x.reshape(B*n, n, d)
            row_valid = valid.reshape(B*n, n)
            row_pad = ~row_valid
            row_out = self.rows[layer](row_x, src_key_padding_mask=row_pad)
            row_out = row_out.reshape(B, n, n, d)

            col_x = x.permute(0, 2, 1, 3).reshape(B*n, n, d)
            col_valid = valid.permute(0, 2, 1).reshape(B*n, n)
            col_pad = ~col_valid
            col_out = self.cols[layer](col_x, src_key_padding_mask=col_pad)
            col_out = col_out.reshape(B, n , n, d).permute(0, 2, 1, 3)

            x = self.fuse[layer](torch.cat([row_out, col_out], dim=-1))

        y = self.out_proj(x)
        P_hat = y[..., 0]
        Q_hat = y[..., 1]

        return P_hat, Q_hat
        

class DenseBlock(nn.Module):    
    def __init__(self, in_features, out_features, act_fn):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = act_fn()
        
    def forward(self, x):              
        y = self.act_fn(self.fc(x))
        return y

    
class OutputBlock(nn.Module):
    def __init__(self, latent_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.fc_r = nn.Linear(latent_dim, (self.num_agents + 1) * self.num_agents)
        self.fc_c = nn.Linear(latent_dim, self.num_agents * (self.num_agents + 1))

    def forward(self, x, mask_p, mask_q):
        
        row = self.fc_r(x).view(-1, self.num_agents + 1, self.num_agents)
        col = self.fc_c(x).view(-1, self.num_agents, self.num_agents + 1)

        row = F.softplus(row) * mask_p
        col = F.softplus(col) * mask_q
        
        row = F.normalize(row, p = 1, dim = 1, eps=1e-8)[:, :-1, :]
        col = F.normalize(col, p = 1, dim = 2, eps=1e-8)[:, :, :-1]

        return torch.minimum(row, col)

    
class MLPHead(nn.Module):
    def __init__(self, net_arch, act_fn, num_agents):
        super().__init__()
        
        blocks = []
        self.num_agents = num_agents
        last_layer_dim = 2 * (self.num_agents**2)
        
        for curr_layer_dim in net_arch:
            blocks.append(DenseBlock(last_layer_dim, curr_layer_dim, act_fn))
            last_layer_dim = curr_layer_dim
              
        self.network = nn.Sequential(*blocks)
        self.output = OutputBlock(last_layer_dim, self.num_agents)             

        
    def forward(self, p, q, mask_p, mask_q):
        x = torch.stack([p, q], dim = 1).view(-1, 2 * (self.num_agents**2))
        x = self.network(x)

        return self.output(x, mask_p, mask_q)


class Mechanism(nn.Module):
    def __init__(self, d_model, n_layer, n_head, dim_ffm, net_arch, act_fn, num_agents):
        super().__init__()

        self.transformer = MatchingTransformer(num_agents, d_model, n_layer, n_head, dim_ffm)
        self.head = MLPHead(net_arch, act_fn, num_agents)
    
    def forward(self, P, Q):
        mask_p = torch.nn.functional.pad((P > 0).to(P.dtype), (0, 0, 0, 1, 0, 0), mode='constant', value=1)
        mask_q = torch.nn.functional.pad((Q > 0).to(Q.dtype), (0, 1, 0, 0, 0, 0), mode='constant', value=1)

        P, Q = self.transformer(P, Q)
        return self.head(P, Q, mask_p, mask_q)

"""
from data_gen import Data

device = "mps"
data = Data(10, 0.5, 0.5)
model = Mechanism(
    64, 4, 2, 256,
    [256, 256, 256, 256, 256], nn.LeakyReLU, 10
).to(device)

p, q = data.generate_batch(64)
P = torch.tensor(p, dtype=torch.float32, device="mps")
Q = torch.tensor(q, dtype=torch.float32, device="mps")

R = model(P, Q)

"""
