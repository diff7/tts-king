import os
import shutil

# traverse root directory, and list directories as dirs and files as files

SOURCE_DIR = "./dataset_main/speakers/amed_shaman/"

# russian_single
# noname_opentts

csv_records = []
for file in os.listdir(SOURCE_DIR):
    if ".lab" in file:
        txt_path = os.path.join(SOURCE_DIR, file)
        with open(txt_path, "r") as f:
            text = f.read().replace("\n", "")
        string = f"{file.replace('.txt','')}|{text}|{text}"
        csv_records.append(string)

final_path = os.path.join(SOURCE_DIR, "metadata.csv")
with open(final_path, "w") as f:
    for text in set(csv_records):
        f.write(text + "\n")
