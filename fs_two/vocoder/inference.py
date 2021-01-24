from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from vocoder.env import AttrDict
from vocoder.meldataset import MAX_WAV_VALUE
from vocoder.models import Generator


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def load_hifigan(model_path, config_path, device):
    with open(config_path) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(model_path, device)
    generator.load_state_dict(state_dict_g['generator'])
    return generator
