from string import ascii_letters, digits, whitespace

cyrillic_letters = (
    "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
)


def strip(text):
    allowed_chars = cyrillic_letters  # + digits + whitespace
    return "".join([c for c in text if c in allowed_chars])


with open("vocab.lab", "r") as r:
    lines = r.read()
lines = sorted([strip(l) for l in lines.split("\n")], key=len)

with open("./vocab_clean.txt", "w") as f:
    for text in lines:
        f.write(text + "\n")
