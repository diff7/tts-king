import os

import torch
import math as m
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter
import wandb as logger

from tqdm import tqdm
from omegaconf import OmegaConf

from hifiapi import HIFIapi

from fs_two.utils.model import get_model, get_param_num
from fs_two.utils.tools import to_device, log, synth_one_sample
from fs_two.model import FastSpeech2Loss
from fs_two.dataset import Dataset
from fs_two.evaluate import evaluate


def main_train_step(
    model,
    batch,
    step,
    optimizer,
    cfg,
    Loss,
):

    grad_acc_step = cfg.train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = cfg.train_config["optimizer"]["grad_clip_thresh"]

    output = model(*(batch[2:]))

    losses = Loss(batch, output)
    total_loss = losses[0]

    # Backward

    total_loss = total_loss / grad_acc_step
    total_loss.backward()
    losses = [l.item() / grad_acc_step for l in losses[1:]]

    if step % grad_acc_step == 0:
        # Clipping gradients to avoid gradient explosion

        # Update weights
        optimizer.step_and_update_lr()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
        optimizer.zero_grad()

    return losses, output


def train_logger(losses, step, total_step, outer_bar, log, logger):

    losses = [sum(losses)] + losses
    message1 = "Step {}/{}, ".format(step, total_step)
    message2 = """Total Loss: {:.4f},
                Mel Loss: {:.4f},
                Pitch Loss: {:.4f},
                Energy Loss: {:.4f},
                Duration Loss: {:.4f}
                Mean pitch: {:.4f}
                Std pitch: {:.4f}
                """.format(
        *losses
    )

    outer_bar.write(message1 + message2)
    log(logger, "train", step, losses=losses)


def main(cfg):
    print("Prepare training ...")

    device = cfg.gpu
    # Get dataset
    dataset = Dataset(
        "train.txt",
        cfg.preprocess_config,
        cfg.train_config,
        sort=True,
        drop_last=True,
    )
    batch_size = cfg.train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
    )

    # Prepare model
    model, optimizer = get_model(cfg, device, train=True)

    # model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(cfg.preprocess_config, cfg.model_config)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = HIFIapi(cfg, cfg.gpu)

    # Init logger
    for p in cfg.train_config["path"].values():
        os.makedirs(p, exist_ok=True)

    os.environ["WANDB_API_KEY"] = cfg.logger.wandb_key
    if cfg.logger.offline:
        os.environ["WANDB_MODE"] = "offline"

    logger.init(name=cfg.exp_name, project="FS2", reinit=True)

    # Training

    step = cfg.tts.restore_step + 1
    epoch = 1
    total_step = cfg.train_config["step"]["total_step"]
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = cfg.tts.restore_step
    outer_bar.update()

    if cfg.run_debug_eval:
        print("RUN SANITY CHECK EVAL:")
        message = evaluate(model, 0, cfg, logger, "val", vocoder, cfg.gpu)

    while True:
        inner_bar = tqdm(
            total=len(loader), desc="Epoch {}".format(epoch), position=1
        )
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward

                losses, output = main_train_step(
                    model,
                    batch,
                    step,
                    optimizer,
                    cfg,
                    Loss,
                )

                if step % cfg.train_config.step.log_step == 0:
                    train_logger(
                        losses,
                        step,
                        total_step,
                        outer_bar,
                        log,
                        logger,
                    )

                if step % cfg.train_config.step.synth_step == 0:
                    (
                        fig,
                        wav_reconstruction,
                        wav_prediction,
                        tag,
                    ) = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        cfg.model_config,
                        cfg.preprocess_config,
                    )
                    log(
                        logger,
                        "train",
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = cfg.preprocess_config["preprocessing"][
                        "audio"
                    ]["sampling_rate"]
                    log(
                        logger,
                        "train",
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(
                            step, tag
                        ),
                    )
                    log(
                        logger,
                        "train",
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % cfg.train_config.step.val_step == 0:
                    model.eval()
                    message = evaluate(
                        model, step, cfg, logger, "val", vocoder, cfg.gpu
                    )
                    outer_bar.write(message)

                    model.train()

                if step % cfg.train_config.step.save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer._optimizer.optimizer.state_dict(),
                        },
                        os.path.join(
                            cfg.train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":

    configs = OmegaConf.load("./config.yaml")
    main(configs)
