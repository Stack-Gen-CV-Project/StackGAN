"""StackGAN-v2 (StackGAN++) generator architecture for CUB inference.

Ported from hanzhanggit/StackGAN-v2 (code/model.py) with these PyTorch-2.x
modernizations so the official `netG_*.pth` checkpoint loads unchanged:
  - Removed `from torch.autograd import Variable` and Variable() wrapping
  - Replaced `F.sigmoid` with `torch.sigmoid` (deprecation)
  - Made the device check inside CA_NET.reparametrize tensor-derived
  - Inlined the cfg values used by the CUB pretrained config (eval_birds.yml +
    miscc/config.py defaults)

The state-dict key naming is preserved exactly, so torch.load + load_state_dict
on `netG_210000.pth` (Google Drive id 1s5Yf3nFiXx0lltMFOiJWB6s1LP24RcwH) works.
"""

import torch
import torch.nn as nn


class _GANCfg:
    Z_DIM = 100
    EMBEDDING_DIM = 128
    GF_DIM = 64
    R_NUM = 2
    B_CONDITION = True


class _TreeCfg:
    BRANCH_NUM = 3


class _TextCfg:
    DIMENSION = 1024


class _Cfg:
    GAN = _GANCfg
    TREE = _TreeCfg
    TEXT = _TextCfg


cfg = _Cfg


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels don't divide 2"
        nc = nc // 2
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def upBlock(in_planes, out_planes):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU(),
    )


def Block3x3_relu(in_planes, out_planes):
    return nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
        )

    def forward(self, x):
        return x + self.block(x)


class CA_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.gf_dim = ngf
        self.in_dim = (cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM) if cfg.GAN.B_CONDITION else cfg.GAN.Z_DIM
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU(),
        )
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code=None):
        if cfg.GAN.B_CONDITION and c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)
        else:
            in_code = z_code
        out_code = self.fc(in_code).view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        super().__init__()
        self.gf_dim = ngf
        self.ef_dim = cfg.GAN.EMBEDDING_DIM if cfg.GAN.B_CONDITION else cfg.GAN.Z_DIM
        self.num_residual = num_residual
        self.jointConv = Block3x3_relu(ngf + self.ef_dim, ngf)
        self.residual = nn.Sequential(*[ResBlock(ngf) for _ in range(num_residual)])
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c = c_code.view(-1, self.ef_dim, 1, 1).repeat(1, 1, s_size, s_size)
        h_c = torch.cat((c, h_code), 1)
        out = self.jointConv(h_c)
        out = self.residual(out)
        out = self.upsample(out)
        return out


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)


class G_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)

    def forward(self, z_code, text_embedding=None):
        if cfg.GAN.B_CONDITION and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None

        fake_imgs = []
        h_code1 = self.h_net1(z_code, c_code)
        fake_imgs.append(self.img_net1(h_code1))
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_imgs.append(self.img_net2(h_code2))
            if cfg.TREE.BRANCH_NUM > 2:
                h_code3 = self.h_net3(h_code2, c_code)
                fake_imgs.append(self.img_net3(h_code3))
        return fake_imgs, mu, logvar
