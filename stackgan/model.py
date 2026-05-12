import torch
import torch.nn as nn


Z_DIM = 100
EMBEDDING_DIM = 128
GF_DIM = 64
R_NUM = 2
TEXT_DIM = 1024


def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1) // 2
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def upBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv3x3(in_ch, out_ch * 2),
        nn.BatchNorm2d(out_ch * 2),
        GLU(),
    )


def Block3x3_relu(in_ch, out_ch):
    return nn.Sequential(
        conv3x3(in_ch, out_ch * 2),
        nn.BatchNorm2d(out_ch * 2),
        GLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(ch, ch * 2), nn.BatchNorm2d(ch * 2), GLU(),
            conv3x3(ch, ch), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class CA_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(TEXT_DIM, EMBEDDING_DIM * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_emb):
        x = self.relu(self.fc(text_emb))
        return x[:, :EMBEDDING_DIM], x[:, EMBEDDING_DIM:]

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        return torch.randn_like(std).mul(std).add_(mu)

    def forward(self, text_emb):
        mu, logvar = self.encode(text_emb)
        return self.reparametrize(mu, logvar), mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.gf_dim = ngf
        self.fc = nn.Sequential(
            nn.Linear(Z_DIM + EMBEDDING_DIM, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU(),
        )
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        x = torch.cat((c_code, z_code), 1)
        x = self.fc(x).view(-1, self.gf_dim, 4, 4)
        return self.upsample4(self.upsample3(self.upsample2(self.upsample1(x))))


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=R_NUM):
        super().__init__()
        self.jointConv = Block3x3_relu(ngf + EMBEDDING_DIM, ngf)
        self.residual = nn.Sequential(*[ResBlock(ngf) for _ in range(num_residual)])
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s = h_code.size(2)
        c = c_code.view(-1, EMBEDDING_DIM, 1, 1).repeat(1, 1, s, s)
        x = self.jointConv(torch.cat((c, h_code), 1))
        return self.upsample(self.residual(x))


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)


class G_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.ca_net = CA_NET()
        self.h_net1 = INIT_STAGE_G(GF_DIM * 16)
        self.img_net1 = GET_IMAGE_G(GF_DIM)
        self.h_net2 = NEXT_STAGE_G(GF_DIM)
        self.img_net2 = GET_IMAGE_G(GF_DIM // 2)
        self.h_net3 = NEXT_STAGE_G(GF_DIM // 2)
        self.img_net3 = GET_IMAGE_G(GF_DIM // 4)

    def forward(self, z_code, text_emb):
        c_code, mu, logvar = self.ca_net(text_emb)
        h1 = self.h_net1(z_code, c_code)
        h2 = self.h_net2(h1, c_code)
        h3 = self.h_net3(h2, c_code)
        return [self.img_net1(h1), self.img_net2(h2), self.img_net3(h3)], mu, logvar
