# This utils.py file handles the math for Object Detection boxes.

import torch
from math import sqrt as sqrt
from itertools import product as product

class AnchorGenerator(object):
    def __init__(self, img_size):
        self.img_size = img_size
        
        # Fixing the Anchors
        # 24px: Direct match for standard icons
        # 48px: Match for zoomed/clustered icons
        self.min_sizes = [24, 48] 
        
        # Ratios: Square (1.0) and Rectangle (1.5)
        self.aspect_ratios = [1.0, 1.5] 
        
        # Stride Steps: Must match the feature map strides in the model (8 and 16)
        self.steps = [8, 16] 

        # Calculate grid sizes based on image size (512px)
        # 512 / 8  = 64x64 Grid (High Resolution for small objects)
        # 512 / 16 = 32x32 Grid (Medium Resolution)
        self.feature_maps = [
            [img_size // 8, img_size // 8],   
            [img_size // 16, img_size // 16]  
        ]
        self.clip = True

    def forward(self, device):
        mean = [] # The list of anchor boxes.
        for k, f in enumerate(self.feature_maps): # Iterate over both feature map scales (64x64 and 32x32)
            for i, j in product(range(f[0]), range(f[1])): # Iterate over every pixel in the grid
                f_k = self.img_size / self.steps[k]

                # Calculate Center Coordinates (cx, cy)
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # Small Square Box
                s_k = self.min_sizes[k] / self.img_size
                mean += [cx, cy, s_k, s_k]

                # Extra scale (1.25x size)
                s_k_prime = sqrt(s_k * (self.min_sizes[k] * 1.25 / self.img_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                
                # Aspect Ratio Boxes (Rectangles)
                for ar in self.aspect_ratios:
                    if ar == 1.0: continue
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.tensor(mean).view(-1, 4) # Converting list to Tensor
        if self.clip:
            output.clamp_(max=1, min=0) # Keep boxes inside image bounds
        return output.to(device)

def decode(loc, priors, variances=[0.1, 0.2]): # Decodes the model's raw output (offsets) into real bounding boxes.
    boxes = torch.cat(( # Calculate Center (cx, cy) and Predicted Center = Prior Center + (Offset * Variance * Prior Size)
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # Convert (cx, cy, w, h) -> (x_min, y_min, x_max, y_max)
    boxes[:, :2] -= boxes[:, 2:] / 2 # Top-Left
    boxes[:, 2:] += boxes[:, :2] # Bottom-Right
    return boxes

def encode(matched, priors, variances=[0.1, 0.2]): # Encodes real boxes into offsets for training (Inverse of decode).
    # Calculate target offsets for center and size
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    # Return encoded targets
    return torch.cat([g_cxcy, g_wh], 1)
