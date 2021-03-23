import os
import numpy as np
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


import fs_two.audio as Audio


class LTraier(pl.LightningModule):
    def __init__(
        self, cfg, model, loss_fn, scheduled_optim, optimizer, voocoder
    ):
        super().init__()

        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.voocoder = voocoder

    def forward(self, values):
        return self.model(*values)

    def configure_optimizers(self):
        optimizer = self.optimizer
        # scheduler = CosineAnnealingLR(optimizer, T_max=1000)
        return ([optimizer],)  # [scheduler]

    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    #     optimizer.step()
    #     optimizer.zero_grad()

    def one_step(self, batches, validation=True):
        total_losses = []
        losses_dict = dict()
        losses_names = [
            "total_loss",
            "mel_loss",
            "mel_postnet_loss",
            "d_loss",
            "f_loss",
            "e_loss",
        ]

        for l_n in losses_names:
            losses_dict[l_n] = 0
        for i, batch in enumerate(batches):

            for j, data_of_batch in enumerate(batch):

                # Get Data
                # TODO extract speaker embedding
                phonemes = data_of_batch["text"]
                mel_target = data_of_batch["mel_target"]

                btach_items_keys = [
                    "text",
                    "D",
                    "log_D",
                    "f0",
                    "energy",
                    "src_len",
                    "mel_len",
                ]
                batch_items = [data_of_batch[K] for K in btach_items_keys]
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                mel_len = data_of_batch["mel_len"].astype(np.int32)
                batch_items = batch_items + [max_src_len, max_mel_len]

                # Forward
                # TODO pass speaker embedding

                out = self.forward(batch_items)

                (
                    mel_output,
                    mel_postnet_output,
                    log_duration_output,
                    f0_output,
                    energy_output,
                    src_mask,
                    mel_mask,
                    _,
                ) = out

                losses_values = self.loss_fn(
                    log_duration_output,
                    data_of_batch["log_D"],
                    f0_output,
                    data_of_batch["f0"],
                    energy_output,
                    data_of_batch["energy"],
                    mel_output,
                    mel_postnet_output,
                    mel_target,
                    src_mask,
                    mel_mask,
                )

                total_loss = sum(losses_values)
                total_losses.append(total_loss.sum())

                # LOSSES / FOR LOGGER
                # total_loss, mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss
                losses_values_items = [l.item() for l in losses_values]
                for l_name, value in zip(
                    losses_names, [total_loss] + losses_values_items
                ):
                    losses_values[l_name] += value
                losses_values["total_loss"] += total_loss.item()

                if not validation:
                    while (i + 1) * (j + 1) < self.cfg.num_audio_exampls:
                        self.synth_step(
                            self.cfg,
                            mel_len,
                            mel_target,
                            mel_postnet_output,
                            mel_output,
                            self.vocoder,
                        )

        i += 1
        j += 1
        losses_values = [
            losses_values[l_name] / ((i * j)) for l_name in losses_names
        ]

        return sum(total_losses) / (i * j), losses_values

    def training_step(self, batches, batch_nb):
        total_losses, losses_values = self.one_step(batches, validatio=False)
        return {"loss": total_losses, "log": losses_values}

    def validation_step(self, batches, batch_nb):
        total_losses, losses_values = self.one_step(batches, validatio=True)
        return {"val_loss": total_losses, "val_log": losses_values}

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack(
            [x["val_loss"] for x in validation_step_outputs]
        ).mean()

        return {"val_loss": val_loss}

    def synth_step(
        self,
        cfg,
        mel_len,
        mel_target,
        mel_postnet_output,
        current_step,
        mel_output,
        vocoder,
    ):
        print("SYNTETHIZING EXAMPLES")
        length = mel_len[0].item()
        mel_target_torch = (
            mel_target[0, :length].detach().unsqueeze(0).transpose(1, 2)
        )
        mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
        mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
        mel_postnet_torch = (
            mel_postnet_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
        )
        mel_postnet = (
            mel_postnet_output[0, :length].detach().cpu().transpose(0, 1)
        )

        # TODO REVRITE
        Audio.tools.inv_mel_spec(
            mel,
            os.path.join(cfg.results_path),
            "step_{}_griffin_lim.wav".format(current_step),
        ),

        Audio.tools.inv_mel_spec(
            mel_postnet,
            os.path.join(
                cfg.results_path,
                "step_{}_postnet_griffin_lim.wav".format(current_step),
            ),
        )

        speech_gen, speech_true = None, None
        if vocoder is not None:
            speech_gen = np.array(vocoder.generate(mel_postnet_torch))
            speech_true = np.array(vocoder.generate(mel_target_torch))

            self.logger.experiment.log(
                {
                    "examples_audio": [
                        wandb.Audio(
                            speech_gen, caption="Generated", sample_rate=16
                        )
                    ]
                }
            )
            self.logger.experiment.log(
                {
                    "examples_audio": [
                        wandb.Audio(
                            speech_true, caption="Ground truth", sample_rate=16
                        )
                    ]
                }
            )

        self.logger.experiment.log(
            {"examples_mel": [wandb.Image(mel_target, caption="Mel Target")]}
        )
        self.logger.experiment.log(
            {
                "examples_mel": [
                    wandb.Image(speech_true, caption="Mel postnet output")
                ]
            }
        )

        return mel_postnet, length, mel_postnet_torch


def train_fs(
    cfg,
    model,
    loss_fn,
    train_data_loader,
    val_data_loader,
    optimizer,
    voocoder,
    save_weights_dir,
    resume_lighting=None,
):

    os.environ["WANDB_API_KEY"] = cfg.wandb_key
    wandb_logger = WandbLogger(name="FS2", project="TEST")

    os.makedirs(weights_dir, exist_ok=True)

    model_pl = pl.Traier(cfg, model, loss_fn, optimizer, voocoder)

    checkpoint_callback = ModelCheckpoint(
        filepath=save_weights_dir,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )

    if resume_lighting is not None:
        resume_lighting = os.path.join(weights_dir, resume_lighting)
        print(resume_lighting)

    trainer = pl.Trainer(
        resume_from_checkpoint=resume_lighting,
        max_epochs=cfg.epochs,
        gpus=[1],
        auto_select_gpus=True,
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=cfg.validation_interval,
    )

    trainer.fit(model_pl, train_data_loader, val_data_loader)

    return model_pl
