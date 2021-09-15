from torch import distributed as dist
from getpass import getuser
from socket import gethostname
from distutils.version import LooseVersion
import torch

TORCH_VERSION = torch.__version__


def get_dist_info():
    if LooseVersion(TORCH_VERSION) < LooseVersion('1.0'):
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_host_info():
    return f'{getuser()}@{gethostname()}'


def tensor_to_device(data, device):
    for key, value in data.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
            elif isinstance(value, dict):
                tensor_to_device(value, device)

def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port