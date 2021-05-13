import torch
from torch import nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class TransposeBN(nn.Module):
    def __init__(self, dim):
        super(TransposeBN, self).__init__()
        self.bn = nn.BatchNorm1d(dim)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn(x)
        return x.permute(0,2,1)



class ViT(nn.Module):
    def __init__(self, num_classes:int=10, img_size:int=32, patch:int=4, dropout:float=0., num_layers:int=8, hidden:int=384, mlp_hidden:int=384, head:int=12):
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.pos_emb = nn.Parameter(torch.randn(1, (self.patch**2)+1, hidden))
        # self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, head, mlp_hidden, dropout=dropout, activation="gelu"), num_layers, norm=TransposeBN(hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            # nn.LayerNorm(hidden),
            Mul(0.3),
            # nn.BatchNorm1d(hidden),
            nn.Linear(hidden, num_classes), # for cls_token
            Mul(0.3)
        )


    def forward(self, x):
        out = self._to_words(x)
        out = torch.cat([self.cls_token.repeat(out.size(0),1,1), self.emb(out)],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        out = out[:,0]
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out