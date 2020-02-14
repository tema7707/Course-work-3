import torch.utils.data
import torch

import torch.nn as nn

from torch.utils.data import Dataset


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def l2normalize(self, vector, eps = 1e-15):
        return vector/(vector.norm()+eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Fashion_swapper_dataset(Dataset):
    
    def __init__(self, loader, obj=31, transform=None):
        self.objects = obj
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, idx):
        first_name = self.loader['objects'][self.objects][idx]
        first_image = read_image(first_name)
        
        first_mask = read_mask(first_name, self.objects)
        first_mask = Image.fromarray(first_mask)
        first_image = self.transform(first_image)
        first_mask = self.transform(first_mask)
        imagewithmask = add_mask(first_image, first_mask, 0)
        return imagewithmask
    
    def __len__(self):
        return self.loader['objects_count'][self.objects]