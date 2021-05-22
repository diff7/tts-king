import os
import shutil

# traverse root directory, and list directories as dirs and files as files

FINAL_DIR = "./ailabs_speaker"

os.makedirs(FINAL_DIR, exist_ok=True)


def csv_dict(path):
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split("|")
        yield line[0], line[2].lower()


def make_record(f_path, name, text, speaker):
    DIR = os.path.join(FINAL_DIR, speaker)
    os.makedirs(DIR, exist_ok=True)
    destination_wav = os.path.join(DIR, name + ".wav")
    destination_lab = os.path.join(DIR, name + ".lab")
    shutil.copy(f_path, destination_wav)
    with open(destination_lab, "w") as f:
        f.write(text)


texts = []
for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
    if "metadata.csv" in files:
        csv_path = os.path.join(root, "metadata.csv")
        for name, text in csv_dict(csv_path):
            file_path = os.path.join(root, "wavs", name + ".wav")
            speaker = root.split("/")[-2]
            text = text.replace("ё", "йо")
            # make_record(file_path, name, text, speaker)
            texts = texts + text.split(" ")

print(len(set(texts)))
with open("./vocab.lab", "w") as f:
    for text in set(texts):
        f.write(text + "\n")
