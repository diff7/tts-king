from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import os
import torch
from hifi.models import Generator


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def load_hifigan(config, weights_path, device):
    generator = Generator(config).to(device)
    state_dict_g = load_checkpoint(weights_path, device)
    generator.load_state_dict(state_dict_g["generator"])
    return generator
