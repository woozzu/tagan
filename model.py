import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


class ResidualBlock(nn.Module):
    def __init__(self, ndim):
        super(ResidualBlock, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(ndim, ndim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndim, ndim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ndim)
        )

    def forward(self, x):
        return x + self.encoder(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # residual blocks
        self.residual_blocks = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

        # conditioning augmentation
        self.mu = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.apply(init_weights)

    def forward(self, img, txt):
        # image encoder
        e = self.encoder(img)

        # text encoder
        if type(txt) is not tuple:
            raise TypeError('txt must be tuple (txt_data, txt_len).')

        txt_data = txt[0]
        txt_len = txt[1]

        hi_f = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
        hi_b = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt_data.size(0)):
            mask_i = (txt_data.size(0) - 1 - i < txt_len).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt_data[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt_data[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        h = (h_f + h_b) / 2
        cond = h.sum(0) / mask.sum(0)

        z_mean = self.mu(cond)
        z_log_stddev = self.log_sigma(cond)
        z = torch.randn(cond.size(0), 128, device=txt_data.device)
        cond = z_mean + z_log_stddev.exp() * z

        # residual blocks
        cond = cond.unsqueeze(-1).unsqueeze(-1)
        merge = self.residual_blocks(torch.cat((e, cond.repeat(1, 1, e.size(2), e.size(3))), 1))

        # decoder
        d = self.decoder(e + merge)

        return d, (z_mean, z_log_stddev)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.eps = 1e-7

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.GAP_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # text feature
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.gen_filter = nn.ModuleList([
            nn.Linear(512, 256 + 1),
            nn.Linear(512, 512 + 1),
            nn.Linear(512, 512 + 1)
        ])
        self.gen_weight = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(-1)
        )

        self.classifier = nn.Conv2d(512, 1, 4)

        self.apply(init_weights)

    def forward(self, img, txt, len_txt, negative=False):
        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)
        img_feats = [self.GAP_1(img_feat_1), self.GAP_2(img_feat_2), self.GAP_3(img_feat_3)]
        D = self.classifier(img_feat_3).squeeze()

        # text attention
        u, m, mask = self._encode_txt(txt, len_txt)
        att_txt = (u * m.unsqueeze(0)).sum(-1)
        att_txt_exp = att_txt.exp() * mask.squeeze(-1)
        att_txt = (att_txt_exp / att_txt_exp.sum(0, keepdim=True))

        weight = self.gen_weight(u).permute(2, 1, 0)

        sim = 0
        sim_n = 0
        idx = np.arange(0, img.size(0))
        idx_n = torch.tensor(np.roll(idx, 1), dtype=torch.long, device=txt.device)

        for i in range(3):
            img_feat = img_feats[i]
            W_cond = self.gen_filter[i](u).permute(1, 0, 2)
            W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
            img_feat = img_feat.mean(-1).mean(-1).unsqueeze(-1)

            if negative:
                W_cond_n, b_cond_n, weight_n = W_cond[idx_n], b_cond[idx_n], weight[i][idx_n]
                sim_n += torch.sigmoid(torch.bmm(W_cond_n, img_feat) + b_cond_n).squeeze(-1) * weight_n
            sim += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * weight[i]

        if negative:
            att_txt_n = att_txt[:, idx_n]
            sim_n = torch.clamp(sim_n + self.eps, max=1).t().pow(att_txt_n).prod(0)
        sim = torch.clamp(sim + self.eps, max=1).t().pow(att_txt).prod(0)

        if negative:
            return D, sim, sim_n
        return D, sim

    def _encode_txt(self, txt, len_txt):
        hi_f = torch.zeros(txt.size(1), 512, device=txt.device)
        hi_b = torch.zeros(txt.size(1), 512, device=txt.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt.size(0)):
            mask_i = (txt.size(0) - 1 - i < len_txt).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        u = (h_f + h_b) / 2
        m = u.sum(0) / mask.sum(0)
        return u, m, mask
