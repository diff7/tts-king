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
from fs_two.model.gan import Discriminator
from fs_two.dataset import Dataset
from fs_two.evaluate import evaluate


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
    netD = Discriminator(
        inut_dim=cfg.gan.inut_dim,
        num_D=cfg.gan.num_D,
        ndf=cfg.gan.ndf,
        n_layers=cfg.gan.n_layers,
        downsampling_factor=cfg.gan.downsampling_factor,
    ).to(device)

    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    # model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(cfg.preprocess_config, cfg.model_config)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = HIFIapi(cfg, cfg.gpu)

    # Init logger
    for p in cfg.train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(cfg.train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(cfg.train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    os.environ["WANDB_API_KEY"] = cfg.logger.wandb_key
    if cfg.logger.offline:
        os.environ["WANDB_MODE"] = "offline"

    logger.init(name=cfg.exp_name, project="FS2", reinit=True)

    # Training
    step = cfg.tts.restore_step + 1
    epoch = 1
    grad_acc_step = cfg.train_config["optimizer"]["grad_acc_step"]
    descriminator_step = cfg.train_config["descriminator"]["step"]
    descriminator_leg_up = cfg.train_config["descriminator"]["leg_up"]

    grad_clip_thresh = cfg.train_config["optimizer"]["grad_clip_thresh"]
    total_step = cfg.train_config["step"]["total_step"]
    log_step = cfg.train_config["step"]["log_step"]
    save_step = cfg.train_config["step"]["save_step"]
    synth_step = cfg.train_config["step"]["synth_step"]
    val_step = cfg.train_config["step"]["val_step"]

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
                output = model(*(batch[2:]))

                # Train Discriminator
                step_weight = abs(m.sin(step))

                D_fake_det = netD(output[9].detach())
                D_real = netD(batch[6])

                loss_D = 0
                for out in D_fake_det:
                    loss_D += step_weight * (F.relu(1 + out[-1]) ** 2).mean()

                for out in D_real:
                    loss_D += step_weight * (F.relu(1 - out[-1]) ** 2).mean()

                loss_D.backward()
                nn.utils.clip_grad_norm_(netD.parameters(), grad_clip_thresh)
                optD.step()
                optD.zero_grad()

                # LET DESCRIMINATOR OUTPERFORM MODEL AT THE BEGGINING #
                # SLOW DOWN MAIN MODEL A BIT #
                if (step % descriminator_step == 0) or (
                    step < descriminator_leg_up
                ):
                    # Cal Loss
                    D_fake = netD(output[9])
                    loss_G = 0

                    for out in D_fake:
                        loss_G += (
                            step_weight * (F.relu(1 + out[-1]) ** 2).mean()
                        )

                    loss_feat = 0
                    feat_weights = 4.0 / (cfg.gan.n_layers + 1)
                    D_weights = 1.0 / cfg.gan.num_D
                    wt = D_weights * feat_weights
                    for i in range(cfg.gan.num_D):
                        for j in range(len(D_fake[i]) - 1):
                            # print(D_fake[i][j].shape, D_real[i][j].shape)
                            loss_feat += wt * F.l1_loss(
                                D_fake[i][j], D_real[i][j].detach()
                            )

                    loss_G = 0.1 * step_weight * loss_feat
                    losses = Loss(batch, output)
                    total_loss = losses[0]

                    # Backward
                    gra_div = max(
                        grad_acc_step * int(step < descriminator_leg_up), 1
                    )

                    total_loss = (total_loss / gra_div) + (loss_G / gra_div)
                    total_loss.backward()
                    losses = [l / gra_div for l in losses[1:]]

                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion

                    # Update weights
                    optimizer.update_lr()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_thresh
                    )
                    optimizer.real_step()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    losses = [sum(losses)] + losses + [loss_G] + [loss_D]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = """Total Loss: {:.4f}, 
                                Mel Loss: {:.4f}, 
                                Pitch Loss: {:.4f}, 
                                Energy Loss: {:.4f}, 
                                Duration Loss: {:.4f}
                                Loss_G: {:.4f},
                                Loss_D: {:.4f},
                                """.format(
                        *losses
                    )

                    with open(
                        os.path.join(train_log_path, "log.txt"), "a"
                    ) as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(logger, "train", step, losses=losses)

                if step % synth_step == 0:
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

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(
                        model, step, cfg, logger, "val", vocoder, cfg.gpu
                    )
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
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
