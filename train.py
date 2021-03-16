import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb as logger

from tqdm import tqdm
from omegaconf import OmegaConf

from fs_two.utils.model import get_model, get_vocoder, get_param_num
from fs_two.utils.tools import to_device, log, synth_one_sample
from fs_two.model import FastSpeech2Loss
from fs_two.dataset import Dataset

from fs_two.evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(cfg, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(cfg, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    os.environ["WANDB_API_KEY"] = cfg.wandb_key

    logger.init(name="FS2", project="TEST", reinit=True)

    # Training
    step = cfg.tts.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = cfg.tts.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(
            total=len(loader), desc="Epoch {}".format(epoch), position=1
        )
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                #total_loss = losses[0]

                # Backward
                #total_loss = total_loss / grad_acc_step
                # total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    # nn.utils.clip_grad_norm_(
                    #     model.parameters(), grad_clip_thresh
                    # )

                    # Update weights
                    optimizer.step_and_update_lr(losses)
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    losses = [sum(losses)] + losses
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Mel Loss: {:.4f}, Total Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
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
                        model_config,
                        preprocess_config,
                    )
                    log(
                        logger,
                        "train",
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
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
                        tag="Training/step_{}_{}_synthesized".format(
                            step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(
                        model, step, configs, logger, "val", vocoder
                    )
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
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

    def get_fsp_configs(
        config_folder,
        config_names=["preprocess.yaml", "model.yaml", "train.yaml"],
    ):
        configs = []
        for name in config_names:
            full_path = os.path.join(config_folder, name)
            configs.append(OmegaConf.load(full_path))
        return configs

    configs = get_fsp_configs("./multi_config/")
    cfg = OmegaConf.load("./config.yaml")

    main(cfg, configs)
