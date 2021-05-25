import argparse
import os


import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from fs_two.utils.model import get_model, get_vocoder
from fs_two.utils.tools import to_device, log, synth_one_sample
from fs_two.model import FastSpeech2Loss
from fs_two.dataset import Dataset

# TODO SET device via config


def evaluate(
    model, step, cfg, logger=None, train_val="val", vocoder=None, device=0
):
    # Get dataset
    dataset = Dataset(
        "val.txt",
        cfg.preprocess_config,
        cfg.train_config,
        sort=False,
        drop_last=False,
    )
    batch_size = cfg.train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(cfg.preprocess_config, cfg.model_config)

    # Evaluation
    loss_sums = [0 for _ in range(4)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(1, len(losses)):
                    loss_sums[i - 1] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    loss_means = [sum(loss_means)] + loss_means
    loss_logs = [step] + loss_means

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *loss_logs
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            cfg.model_config,
            cfg.preprocess_config,
        )

        log(logger, "val", step, losses=loss_means)
        log(
            logger,
            "val",
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = cfg.preprocess_config["preprocessing"]["audio"][
            "sampling_rate"
        ]
        log(
            logger,
            "val",
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            "val",
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":
    device = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        required=True,
        help="path to model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        required=True,
        help="path to train.yaml",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(args.model_config, "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(args.train_config, "r"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)
