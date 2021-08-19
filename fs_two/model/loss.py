import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"][
            "energy"
        ]["feature"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        speaker_targets = inputs[2]
        (
            mel_targets,
            _,
            _,
            energy_targets,
            duration_targets,
            pitches_cwt,
            pitch_mean,
            pitch_std,
        ) = inputs[6:]
        (
            mel_predictions,
            pitch_cwt_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            postnet_mel_predictions,
            pitch_mean_pred,
            pitch_std_pred,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, : mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitches_cwt.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        pitch_mask = src_masks.unsqueeze(2)
        pitch_mask = pitch_mask.repeat(1, 1, 11)
        pitch_predictions = pitch_cwt_predictions.masked_select(pitch_mask)
        pitch_targets = pitches_cwt.masked_select(pitch_mask)

        energy_predictions = energy_predictions.masked_select(src_masks)
        energy_targets = energy_targets.masked_select(src_masks)

        log_duration_predictions = log_duration_predictions.masked_select(
            src_masks
        )
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions * mel_masks.unsqueeze(-1)
        # mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions * mel_masks.unsqueeze(
            -1
        )
        # postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))

        # mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets * mel_masks.unsqueeze(-1)

        mel_loss = self.mse_loss(mel_predictions, mel_targets)
        mel_loss_mae = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        total_mel_loss = mel_loss + mel_loss_mae + postnet_mel_loss

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)

        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(
            log_duration_predictions, log_duration_targets
        )

        std_pitch_loss = self.mse_loss(pitch_std_pred, pitch_std.unsqueeze(1))
        mean_pitch_loss = self.mse_loss(
            pitch_mean_pred, pitch_mean.unsqueeze(1)
        )

        total_loss = (
            total_mel_loss
            + duration_loss
            + pitch_loss
            + energy_loss
            + mean_pitch_loss
            + std_pitch_loss
        )

        return (
            total_loss,
            total_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            mean_pitch_loss,
            std_pitch_loss,
        )
