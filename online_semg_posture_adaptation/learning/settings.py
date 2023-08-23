"""
This module defines the general settings for using the PyTorch model(s).
"""

import os

import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDX_DEVICE_STR = '0'

SEED = 1


def set_reproducibility(seed: int = SEED) -> None:

    # PyTorch
    torch.manual_seed(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUDA
    torch.cuda.manual_seed(seed)  # if multi-GPU, use manual_seed_all(seed)

    return


def set_visible_device(idx_device_str: str = IDX_DEVICE_STR) -> None:

    """
    Set the GPU visible by the Python script or notebook.
    Only one visible.
    """

    assert isinstance(idx_device_str, str)
    assert len(idx_device_str) == 1
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = idx_device_str

    return


def main() -> None:
    pass


if __name__ == '__main__':
    main()
