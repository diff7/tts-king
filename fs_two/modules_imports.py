import os
import torch
import numpy as np
import torch.nn as nn

import audio as Audio

from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from optimizer import ScheduledOptim, Ranger
from pcgrad import PCGrad
from evaluate import evaluate
import utils



class FSTWOTrainable:
    def __init__(self, config, weights_path=None, device="gpu"):

        self.model = nn.DataParallel(FastSpeech2()).to(device)
        # Load checkpoint if exists
        self.weights_path = weights_path
        if weights_path is not None:
            checkpoint = torch.load(weights_path)
            self.model.load_state_dict(checkpoint["model"])
        
        self.cfg = config
        self.device = device

        # TODO get the righ restore step
        self.restore_step = 0 

    def train(self, data_loader, loss_fn, voocoder=None, logger=None):
        
        optimizer = PCGrad(
        Ranger(
            self.model.parameters(),
            betas = self.cfg.betas,
            eps = self.cfg.eps,
            weight_decay = self.cfg.weight_decay,
        )
    )
        scheduled_optim = ScheduledOptim(
        self.cfg.decoder_hidden, self.cfg.n_warm_up_step, self.restore_step
    )
        self.model.train()
        model_train(self.cfg, 
                    self.model, 
                    loss_fn, 
                    data_loader, 
                    scheduled_optim, 
                    optimizer, 
                    logger,
                    voocoder,
                    self.device)

        return self.model

    def generate(self, phonemes, duration_control=1.0, pitch_control=1.0, energy_control=1.0, speaker=None):

        self.model.eval()        
        src_len = torch.from_numpy(np.array([phonemes.shape[1]])).to(self.device)
        result = self.model(
            phonemes, src_len, d_control=duration_control, p_control=pitch_control, e_control=energy_control, speaker_emb=speaker)

        # mel, mel_postnet, log_duration_output, f0_output, energy_output
        return result 


def model_train(cfg, 
                model, 
                loss_fn, 
                data_loader, 
                scheduled_optim, 
                optimizer, 
                logger,
                voocoder,
                device="gpu"):

    # INIT LOSSES
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

    for epoch in range(cfg.epochs):
        # Get Training data_loader
        total_step = cfg.epochs * len(data_loader) * cfg.batch_size

        for i, batchs in enumerate(data_loader):
            for j, data_of_batch in enumerate(batchs):

                current_step = (
                    i * cfg.batch_size
                    + j
                    + epoch * len(data_loader) * cfg.batch_size
                    + 1
                )

                # Get Data
                # TODO extract speaker embedding
                phonemes = torch.from_numpy(data_of_batch["text"]).long().to(device)
                mel_target = (
                    torch.from_numpy(data_of_batch["mel_target"])
                    .float()
                    .to(device)
                )
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = (
                    torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                )
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = (
                    torch.from_numpy(data_of_batch["energy"]).float().to(device)
                )
                src_len = (
                    torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                )
                mel_len = (
                    torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                )
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

                # Forward
                # TODO pass speaker embedding
                (
                    mel_output,
                    mel_postnet_output,
                    log_duration_output,
                    f0_output,
                    energy_output,
                    src_mask,
                    mel_mask,
                    _,
                ) = model(
                    phonemes,
                    src_len,
                    mel_len,
                    D,
                    f0,
                    energy,
                    max_src_len,
                    max_mel_len,
                )

                # Cal Loss
                # mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss
                losses_values = loss_fn(
                    log_duration_output,
                    log_D,
                    f0_output,
                    f0,
                    energy_output,
                    energy,
                    mel_output,
                    mel_postnet_output,
                    mel_target,
                    ~src_mask,
                    ~mel_mask,
                )
                
                
                with torch.no_grad():
                    total_loss = sum(losses_values)

                # LOSSES / FOR LOGGER
                # total_loss, mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss

                losses_values_items = [l.item() for l in losses_values]
                for l_n, val in zip(losses_names, [total_loss] + losses_values_items):
                    losses_values[l_n] = val
                losses_values['total_loss'] = total_loss.item()

                # TODO add logger
                # logger.log(losses)

                # Backward
                # total_loss = total_loss / cfg.acc_steps
                # total_loss.backward()

                if current_step % cfg.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip_thresh
                )

                # Update weights
                scheduled_optim.step_and_update_lr(*losses_values)
                scheduled_optim.zero_grad()

                # Print
                if current_step % cfg.log_step == 0:
                    if logger is not None:
                        logging_step(cfg, epoch, losses_dict, logger, current_step, total_step)
                    

                if current_step % cfg.save_step == 0:
                    save_model(model, optimizer, current_step)
                   

                if current_step % cfg.synth_step == 0:
                    mel_postnet, length = synth_step(cfg, mel_len, mel_target, mel_postnet_output, current_step, mel_output, voocoder)
                    save_plots(cfg, f0, f0_output, length, energy_output, energy, mel_postnet, mel_target, current_step)
                    

                if current_step % cfg.eval_step == 0:
                    if logger is not None:
                        eval_model_step(model, logger, current_step, voocoder)


def logging_step(cfg, epoch, losses_dict, logger, current_step, total_step):
    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
        epoch + 1, cfg.epochs, current_step, total_step
    )
    str2 = ' '.join([f'{n} : {i[n]}' for n in losses_dict])
   
    print("\n" + str1)
    print(str2)

    for l_n in losses_dict:
        logger.add_scalar(
        f"Loss/{l_n}", losses_dict[l_n], current_step
    )
    

def save_model(model, optimizer, current_step):
    torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    os.path.join(
        cfg.results_path,
        "checkpoint_{}.pth.tar".format(current_step),
    ),)
    print("save model at step {} ...".format(current_step))

def eval_model_step(model, logger, current_step, voocoder):
    model.eval()
    with torch.no_grad():
        d_l, f_l, e_l, m_l, m_p_l = evaluate(model, current_step, voocoder)
        t_l = d_l + f_l + e_l + m_l + m_p_l

        logger.add_scalar("Loss/total_loss", t_l, current_step)
        logger.add_scalar("Loss/mel_loss", m_l, current_step)
        logger.add_scalar("Loss/mel_postnet_loss", m_p_l, current_step)
        logger.add_scalar("Loss/duration_loss", d_l, current_step)

        logger.add_scalar("Loss/F0_loss", f_l, current_step)
        logger.add_scalar("Loss/energy_loss", e_l, current_step)

    model.train()


def synth_step(
    cfg,
    mel_len,
    mel_target,
    mel_postnet_output,
    current_step,
    mel_output,
    voocoder
):
    length = mel_len[0].item()
    mel_target_torch = (
        mel_target[0, :length].detach().unsqueeze(0).transpose(1, 2)
    )
    mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
    mel_torch = mel_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
    mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
    mel_postnet_torch = (
        mel_postnet_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
    )
    mel_postnet = mel_postnet_output[0, :length].detach().cpu().transpose(0, 1)
    Audio.tools.inv_mel_spec(
        mel,
        os.path.join(
            cfg.results_path),
            "step_{}_griffin_lim.wav".format(current_step),
        ),
    )
    Audio.tools.inv_mel_spec(
        mel_postnet,
        os.path.join(
            cfg.results_path,
            "step_{}_postnet_griffin_lim.wav".format(current_step),
        ),
    )

    if voocoder is not None:
        if cfg.voocoder == "melgan":
            utils.melgan_infer(
                mel_torch,
                voocoder,
                os.path.join(
                    cfg.results_path,
                    "step_{}_{}.wav".format(current_step, cfg.voocoder),
                ),
            )
            utils.melgan_infer(
                mel_postnet_torch,
                voocoder,
                os.path.join(
                    cfg.results_path,
                    "step_{}_postnet_{}.wav".format(current_step, cfg.voocoder),
                ),
            )
            utils.melgan_infer(
                mel_target_torch,
                voocoder,
                os.path.join(
                    cfg.results_path,
                    "step_{}_ground-truth_{}.wav".format(current_step, cfg.voocoder),
                ),
            )
        elif cfg.vocoder == "waveglow":
            utils.waveglow_infer(
                mel_torch,
                voocoder,
                os.path.join(
                    cfg.results_path,
                    "step_{}_{}.wav".format(current_step, cfg.vocoder),
                ),
            )
            utils.waveglow_infer(
                mel_postnet_torch,
                voocoder,
                os.path.join(
                    cfg.results_path,
                    "step_{}_postnet_{}.wav".format(current_step, cfg.vocoder),
                ),
            )
            utils.waveglow_infer(
                mel_target_torch,
                voocoder,
                os.path.join(
                    cfg.results_path,
                    "step_{}_ground-truth_{}.wav".format(current_step, cfg.voocoder),
                ),
            )

    return mel_postnet, length

def save_plots(cfg, f0, f0_output, length, energy_output, energy, mel_postnet, mel_target, current_step):
    f0 = f0[0, :length].detach().cpu().numpy()
    energy = energy[0, :length].detach().cpu().numpy()
    f0_output = f0_output[0, :length].detach().cpu().numpy()
    energy_output = energy_output[0, :length].detach().cpu().numpy()

    utils.plot_data(
        [
            (mel_postnet.numpy(), f0_output, energy_output),
            (mel_target.numpy(), f0, energy),
        ],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
        filename=os.path.join(cfg.results_path, "step_{}.png".format(current_step)),
    )
