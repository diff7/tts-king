import torch
import numpy as np
import torch.nn as nn

from fs_two.model import FastSpeech2
from fs_two.model.loss import FastSpeech2Loss
from fs_two.model.optimizer import ScheduledOptim, Ranger
from train_fs_lighting import train_fs


class FSTWOapi:
    def __init__(self, config, weights_path=None, device=0):

        self.model = nn.DataParallel(FastSpeech2(), device_ids=[device]).to(
            device
        )
        # Load checkpoint if exists
        self.weights_path = weights_path
        if weights_path is not None:
            checkpoint = torch.load(weights_path)
            self.model.load_state_dict(checkpoint["model"])

        self.cfg = config
        self.device = device

        # TODO get the righ restore step
        self.restore_step = 0

    def train(
        self, train_data_loader, val_data_loader, voocoder=None, logger=None
    ):
        loss_fn = FastSpeech2Loss().to(self.device)
        optimizer = Ranger(
            self.model.parameters(),
            betas=self.cfg.betas,
            eps=self.cfg.eps,
            weight_decay=self.cfg.weight_decay,
        )

        # TODO ADD _update_learning_rate to Lighting
        # optimizer = ScheduledOptim(
        #     optimizer, cfg.decoder_hidden, cfg.n_warm_up_step, cfg.restore_step
        # )

        train_fs(
            self.cfg,
            self.model,
            loss_fn,
            train_data_loader,
            val_data_loader,
            optimizer,
            logger,
            voocoder,
            self.device,
            self.cfg.save_weights_dir,
            self.cfg.resume_lighting,
        )

        return self.model

    def generate(
        self,
        phonemes,
        duration_control=1.0,
        pitch_control=1.0,
        energy_control=1.0,
        speaker=None,
    ):
        self.model.eval()
        src_len = torch.from_numpy(np.array([phonemes.shape[1]])).to(
            self.device
        )
        result = self.model(
            phonemes,
            src_len,
            d_control=duration_control,
            p_control=pitch_control,
            e_control=energy_control,
            # speaker_emb=speaker,
        )

        # mel, mel_postnet, log_duration_output, f0_output, energy_output
        return result
