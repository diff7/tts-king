import os
import string

from string import ascii_letters, digits, whitespace

cyrillic_letters = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
)


def my_strip(text):
    allowed_chars = cyrillic_letters + digits + whitespace
    return "".join([c for c in text if c in allowed_chars]).replace("\n", "")


# traverse root directory, and list directories as dirs and files as files

SOURCE_DIR = "./dataset_main/speakers/"


class SpeakerStat:
    def __init__(self):
        self.speakers = dict()

    def add(self, name):
        self.speakers[name] = [0, 0, ""]

    def update(self, name, text):
        len_words = len(text.split(" "))
        self.speakers[name][0] += 1
        self.speakers[name][1] += len_words
        self.speakers[name][2] += " " + text

    def make_csv(self, file_path):
        csv_records = ["source_name|speaker_id|num_sentences|len_words"]
        for speaker in self.speakers:
            dataset_name = speaker.split("_")[-1]
            num_sentences = self.speakers[speaker][0]
            len_words = self.speakers[speaker][1]
            string = f"{dataset_name}|{speaker}|{num_sentences}|{len_words}"
            csv_records.append(string)

        self.save(file_path, csv_records)

    def save(self, file_path, records):
        with open(file_path, "w") as f:
            for text in records:
                f.write(text + "\n")

    def save_vocab(self, file_path):
        words = []
        for speaker in self.speakers:
            sp_words = self.speakers[speaker][2].split(" ")
            sp_words = [w for w in sp_words if len(w) > 0]
            words += sp_words
        words = list(set(words))
        words = sorted(words, key=len)
        print(f"unique words: {len(words)}")
        self.save(file_path, words)


def csv_dict(path):
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split("|")
        if len(line) == 3:
            yield line[0], line[2].lower()
        if len(line) == 2:
            yield line[0], line[1].lower()


def make_record(f_path, text):
    with open(f_path, "w") as f:
        f.write(text)


# def clean(s):
#     exclude = set(
#         list(string.punctuation) + ["", "_", "\n", "...", "..", "«", "»"]
#     )
#     return my_strip("".join(ch for ch in s if ch not in exclude))


def main():
    speakers_lib = SpeakerStat()

    for directory in os.listdir(SOURCE_DIR):
        full_directory = os.path.join(SOURCE_DIR, directory)
        speakers_lib.add(directory)
        csv_path = os.path.join(full_directory, "metadata.csv")
        for name, text in csv_dict(csv_path):
            text = my_strip(text)
            speakers_lib.update(directory, text)
            file_path = os.path.join(full_directory, name + ".txt")
            make_record(file_path, text)
            make_record(file_path.replace("txt", "lab"), text)

    speakers_lib.make_csv("./speaker_stats.csv")
    speakers_lib.save_vocab("./vocab.lab")


if __name__ == "__main__":
    main()
