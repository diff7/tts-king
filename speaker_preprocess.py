from resemblyzer import VoiceEncoder, preprocess_wav
import os
import numpy as np

fpath = Path("path_to_an_audio_file")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
