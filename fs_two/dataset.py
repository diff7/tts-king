import json
import math
import os

import random
import numpy as np
from torch.utils.data import Dataset

from fs_two.text import text_to_sequence
from fs_two.utils.tools import pad_1D, pad_2D
from fs_two.text.symbols import _mask, _silences


def random_mask(text, _silences, max_masks_per_sentence, _mask):
    # randonly mask some sentences
    # we do not want to mask short sentences

    text = text.split(" ")
    max_len = len(text)
    masks_count = int(
        max_masks_per_sentence * max_len
    )  # max_masks_per_sentence = 0.15
    if masks_count == 0:
        return text
    mask_indexes = random.choices(list(range(max_len)), k=masks_count)
    for ind in mask_indexes:
        if not text[ind] in _silences:
            text[ind] = _mask
    return " ".join(text)


class Dataset(Dataset):
    def __init__(
        self,
        filename,
        preprocess_config,
        train_config,
        sort=False,
        drop_last=True,
    ):
        self._silences = [s.replace("@", "") for s in _silences]
        self.max_masks_per_sentence = train_config.max_masks_per_sentence
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"][
            "text_cleaners"
        ]
        self.batch_size = train_config["optimizer"]["batch_size"]

        (
            self.basename,
            self.speaker,
            self.text,
            self.raw_text,
        ) = self.process_meta(filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)

        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        pitch_cwt_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-cwt-pitch-{}.npy".format(speaker, basename),
        )

        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )

        pitch_raw = np.load(pitch_path)
        pitch_cwt = np.load(pitch_cwt_path)

        pitch_mean_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-mean-{}.npy".format(speaker, basename),
        )
        pitch_mean = np.load(pitch_mean_path)

        pitch_std_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-std-{}.npy".format(speaker, basename),
        )
        pitch_std = np.load(pitch_std_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "energy": energy,
            "duration": duration,
            "pitch_raw": pitch_raw,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_cwt": pitch_cwt,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename),
            "r",
            encoding="utf-8",
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                if self.max_masks_per_sentence > 1:
                    t = random_mask(
                        t, self._silences, self.max_masks_per_sentence, _mask
                    )
                text.append(t)
                raw_text.append(r)

            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]

        pitches_mean = [data[idx]["pitch_mean"] for idx in idxs]
        pitches_std = [data[idx]["pitch_std"] for idx in idxs]
        pitches_cwt = [data[idx]["pitch_cwt"] for idx in idxs]
        pitches_raw = [data[idx]["pitch_raw"] for idx in idxs]

        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        pitches_mean = np.array(pitches_mean)
        pitches_std = np.array(pitches_std)

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        energies = pad_1D(energies)
        pitches_raw = pad_1D(pitches_raw)
        durations = pad_1D(durations)

        pitches_cwt = pad_2D(pitches_cwt)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            energies,
            durations,
            pitches_raw,
            pitches_cwt,
            pitches_mean,
            pitches_std,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
