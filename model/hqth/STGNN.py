

'''模型设置要求'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Inverted Temporal Block
# -----------------------------
class InvertedTemporalBlock(nn.Module):
    def __init__(self, hidden_dim, attn_heads=4):
        super().__init__()
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 3 * hidden_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 4, num_heads=attn_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 4)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, T, H]
        w = F.softmax(self.weight_mlp(x), dim=-1)  # [B, T, 1]
        x_cat = torch.cat([w, x], dim=-1)          # [B, T, H+1]

        attn_out, _ = self.attn(x_cat, x_cat, x_cat)  # [B, T, H+1]
        x_norm = self.norm1(attn_out + x_cat)

        ffn_out = self.ffn(x_norm)  # [B, T, H]
        return self.norm2(ffn_out + x)  # residual


# -----------------------------
# Double-Path Fusion
# -----------------------------
def double_direction_fusion(z_inv, z):
    # z_inv, z: [B, T, H]
    attn_temp = F.softmax(torch.matmul(z, z_inv.transpose(-1, -2)) / z.shape[-1]**0.5, dim=-1)
    attn_feat = F.softmax(torch.matmul(z_inv, z.transpose(-1, -2)) / z.shape[-1]**0.5, dim=-1)

    h_temp = torch.matmul(attn_temp, z_inv)
    h_feat = torch.matmul(attn_feat, z)
    return h_temp, h_feat


# -----------------------------
# N-Neighbor Correlation Attention
# -----------------------------
class NNeighborCorrelationAttention(nn.Module):
    def __init__(self, dim, top_k=10):
        super().__init__()
        self.top_k = top_k
        self.qkv_proj = nn.Linear(dim, dim * 3)

    def forward(self, h, z_inv):
        # h: [B, T, H]
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        v = v + z_inv  # bring in z_inv as additional context

        sim = torch.matmul(q, k.transpose(-1, -2)) / q.shape[-1]**0.5
        topk_val, topk_idx = torch.topk(sim, self.top_k, dim=-1)
        mask = torch.zeros_like(sim)
        mask.scatter_(-1, topk_idx, 1.0)
        sim = sim.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(sim, dim=-1)

        return torch.matmul(attn, v)  # [B, T, H]


# -----------------------------
# Double-path Gating Mechanism
# -----------------------------
class DoublePathGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_feat = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.gate_temp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.fuse_gate = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())

    def forward(self, feat, temp):
        gf = self.gate_feat(feat)
        gt = self.gate_temp(temp)
        m = self.fuse_gate(torch.cat([feat, temp], dim=-1))  # [B, T, 1]
        return gf * feat * m + gt * temp * (1 - m)


# -----------------------------
# Decoder with Decomposed Fitting
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, dim, pred_len):
        super().__init__()
        self.agg_weight = nn.Sequential(nn.Linear(dim, 1), nn.Softmax(dim=1))
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, pred_len * 2)
        )
        self.pred_len = pred_len

    def forward(self, x):
        w = self.agg_weight(x)  # [B, T, 1]
        x_agg = torch.sum(x * w, dim=1)  # [B, H]
        out = self.proj(x_agg)           # [B, pred_len * 2]
        mean, dev = out.chunk(2, dim=-1)
        return mean + torch.tanh(dev)    # [B, pred_len]




from kan import KAN
class STGNN(nn.Module):
    def __init__(self, factor_num, hidden_dim=256, pred_len=1, top_k=1, attn_heads=4):
        super(STGNN, self).__init__()


        input_dim = factor_num

        # 输入层
        self.input_layter = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # 隐含层
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 混合层
        input_dim = 128
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.temporal_block = InvertedTemporalBlock(hidden_dim, attn_heads)
        self.n_attn_feat = NNeighborCorrelationAttention(hidden_dim, top_k)
        self.n_attn_temp = NNeighborCorrelationAttention(hidden_dim, top_k)
        self.gate = DoublePathGate(hidden_dim)
        # self.decoder = Decoder(hidden_dim, pred_len)


        # 输出层
        self.output_layer = KAN([256,1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, tsdata):

        tsdata = tsdata.float() 
        tsdata = self.input_layter(tsdata)
        tsdata = self.net(tsdata)

        x = tsdata.reshape(tsdata.shape[0], 1, tsdata.shape[1])  # [B, T, H]

        x_proj = self.encoder(x)             # [B, T, H]
        z_inv = self.temporal_block(x_proj)  # [B, T, H]

        h_temp, h_feat = double_direction_fusion(z_inv, x_proj)
        o_feat = self.n_attn_feat(h_feat, z_inv)
        o_temp = self.n_attn_temp(h_temp, z_inv)

        fused = self.gate(o_feat, o_temp)    # [B, T, H]
        fused = fused.reshape(fused.shape[0], fused.shape[-1])
        output = self.output_layer(fused)  # [B, 1]
        return output
    
if __name__ == "__main__":
    model = STGNN(factor_num=1400)
    # print(model)
    x = torch.randn(32, 1400)
    print(f"x.shape: {x.shape}")  # 应该是 [32, 1400]
    output = model(x)
    print(f"output.shape: {output.shape}")  # 应该是 [32, 1]