import torch
from math import sqrt as sqrt
from itertools import product as product

class AnchorGenerator(object):
    def __init__(self, img_size):
        self.img_size = img_size
        
        # --- ANCHOR FIX ---
        # 24px: Direct match for standard icons
        # 48px: Match for zoomed/clustered icons
        self.min_sizes = [24, 48] 
        
        # Ratios: Square (1.0) and Rectangle (1.5)
        self.aspect_ratios = [1.0, 1.5] 
        
        # --- STEPS MUST MATCH MODEL LAYERS (8 and 16) ---
        self.steps = [8, 16] 
        
        self.feature_maps = [
            [img_size // 8, img_size // 8],   # 64x64
            [img_size // 16, img_size // 16]  # 32x32
        ]
        self.clip = True

    def forward(self, device):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                f_k = self.img_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.img_size
                mean += [cx, cy, s_k, s_k]

                # Extra scale (1.25x size)
                s_k_prime = sqrt(s_k * (self.min_sizes[k] * 1.25 / self.img_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                
                # Aspect Ratios
                for ar in self.aspect_ratios:
                    if ar == 1.0: continue
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output.to(device)

def decode(loc, priors, variances=[0.1, 0.2]):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def encode(matched, priors, variances=[0.1, 0.2]):
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)