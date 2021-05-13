import torch
from torch import nn



class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight



class ViT(nn.Module):
    def __init__(self, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0.5, num_layers:int=1, hidden:int=128, mlp_hidden:int=128*4, head:int=4):
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.pos_emb = nn.Parameter(torch.randn(1, (self.patch**2)+1, hidden))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, head, mlp_hidden, dropout=dropout, activation="gelu"), num_layers)
        self.fc = nn.Sequential(
            # nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes), # for cls_token
            Mul(0.01)
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