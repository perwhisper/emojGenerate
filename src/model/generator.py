import torch
import torch.nn as nn
import torch.nn.functional as F

# 残差块
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.in2 = nn.InstanceNorm2d(dim)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

# 多头交叉注意力（融合情感和角色特征）
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, role_embed_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim + role_embed_dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.role_embedding = nn.Embedding(3, role_embed_dim)

    def forward(self, identity_feat, emotion_feat, role_type):
        B, C, H, W = identity_feat.shape
        identity_feat = identity_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        emotion_feat = emotion_feat.unsqueeze(1)

        # 角色嵌入
        role_embed = self.role_embedding(role_type).unsqueeze(1)
        emotion_feat_with_role = torch.cat([emotion_feat, role_embed], dim=-1)

        # 注意力计算
        q = self.q_proj(emotion_feat_with_role).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(identity_feat).reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(identity_feat).reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, 1, C)
        out = self.out_proj(out).squeeze(1)
        return out

# 主生成器
class EmojiGenerator(nn.Module):
    def __init__(self, img_dim=3, au_dim=16, num_res_blocks=9, num_heads=4, role_embed_dim=16):
        super().__init__()
        # 编码层（提取人脸特征）
        self.enc_1 = nn.Conv2d(img_dim, 64, 7, 1, 3)
        self.enc_2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.enc_3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.enc_4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.in_enc = nn.InstanceNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # 残差块（保留细节特征）
        self.res_blocks = nn.Sequential(*[ResBlock(512) for _ in range(num_res_blocks)])

        # 注意力融合（情感+角色+身份）
        self.attention = MultiHeadCrossAttention(dim=512, num_heads=num_heads, role_embed_dim=role_embed_dim)
        self.au_proj = nn.Linear(au_dim, 512)  # AU编码映射到特征维度

        # 解码层（生成最终图片）
        self.dec_1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1)
        self.dec_2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)
        self.dec_3 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.dec_4 = nn.Conv2d(64, img_dim, 7, 1, 3)
        self.in_dec = nn.InstanceNorm2d(256)
        self.tanh = nn.Tanh()

    def forward(self, img, au_code, role_type):
        # 编码
        x = self.relu(self.enc_1(img))
        x = self.relu(self.enc_2(x))
        x = self.relu(self.enc_3(x))
        x = self.relu(self.in_enc(self.enc_4(x)))
        x = self.res_blocks(x)

        # AU映射+注意力融合
        au_feat = self.relu(self.au_proj(au_code))
        attn_feat = self.attention(x, au_feat, role_type)
        attn_feat = attn_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        x = x + attn_feat  # 融合情感特征

        # 解码
        x = self.relu(self.in_dec(self.dec_1(x)))
        x = self.relu(self.dec_2(x))
        x = self.relu(self.dec_3(x))
        x = self.tanh(self.dec_4(x))  # 输出[-1,1]范围，匹配归一化
        return x
