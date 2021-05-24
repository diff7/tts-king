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
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, predictions):
        speaker_targets = inputs[2]
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            postnet_mel_predictions,
            adv_class,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, : mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(
            src_masks
        )
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mse_loss(mel_predictions, mel_targets)
        mel_loss_mae = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mse_loss(postnet_mel_predictions, mel_targets)
        total_mel_loss = mel_loss + mel_loss_mae + postnet_mel_loss

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(
            log_duration_predictions, log_duration_targets
        )

        print("adv_class", adv_class.shape)
        print("speaker_targets", speaker_targets.shape)
        class_loss = self.criterion(adv_class, speaker_targets.unsqueeze(1).expand(-1, adv_class.size(2)))

        total_loss = (
            total_mel_loss
            + duration_loss
            + pitch_loss
            + energy_loss
            + class_loss
        )

        return (
            total_loss,
            total_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
