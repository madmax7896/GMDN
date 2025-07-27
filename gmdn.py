import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# MDN Head
class MDNHead(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_gaussians=12):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim
        self.pi = nn.Linear(input_dim, num_gaussians)
        self.mu = nn.Linear(input_dim, num_gaussians * output_dim)
        self.sigma = nn.Linear(input_dim, num_gaussians * output_dim)
        self.min_sigma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        pi = F.softmax(self.pi(x), dim=1)
        mu = self.mu(x).view(-1, self.num_gaussians, self.output_dim)
        sigma = torch.exp(self.sigma(x)).view(-1, self.num_gaussians, self.output_dim) + self.min_sigma
        return pi, mu, sigma

# SemGCN Layer
class SemGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        res = self.res_proj(x)
        x = self.gcn(x, edge_index)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + res

# Full Model
class SemGCN_MDN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers, num_gaussians):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SemGCNLayer(in_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SemGCNLayer(hidden_dim, hidden_dim))
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.mdn = MDNHead(hidden_dim, output_dim=1, num_gaussians=num_gaussians)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.norm_out(x)
        return self.mdn(x)

# Loss functions
def mdn_loss(y, pi, mu, sigma):
    y = y.view(y.size(0), 1, 1).expand_as(mu)
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(y).sum(-1)
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)
    return -log_sum.mean()

