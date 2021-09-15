from thop import profile
import torch
from copy import deepcopy



def model_info(model, verbose=False, img_size=448):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    img = torch.zeros((1, 3, img_size, img_size), device=next(model.parameters()).device)  # input
    flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
    fs = ', %.1f GFLOPS at image size = %d' % (flops, img_size)  # 640x640 GFLOPS

    return f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}"