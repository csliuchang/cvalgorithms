import torch


def load_pth(pth):
    state_dict = torch.load(pth)["model"]
    model_dict = state_dict
    pass


if __name__ == "__main__":
    path = "C:\\Users\\user1\\Desktop\\convnext_tiny_sa_1k_224_ema.pth"
    load_pth(path)