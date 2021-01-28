import subprocess
import glob

def convert_mp3_to_wav(new_dir,  filename, sr=None):
    filename = filename.split(".mp3")[0]
    if sr:
        return subprocess.call(["ffmpeg", f"-i {filename}.mp3 -ar {sr} {new_dir}/{filename}.wav"])
    else:
        return subprocess.call(["ffmpeg", f"-i {filename}.mp3 {new_dir}/{filename}.wav"])
    

def convert_dataset(dir, new_dir, sr=None):
    for filename in glob.glob(f"{dir}/**/*.mp3"):
        convert_mp3_to_wav(new_dir, filename, sr)


