import os

FOLDER =  '/home/dev/other/fsp/data/dataset_main/speakers/'

BAD_SANTA_LIST =  '/home/dev/other/fsp/data/dataset_main/aligner/prev_unaligned.txt'

def cat(f1, f2):
    return os.path.join(f1,f2)

def make_key(path):
    path = ''.join(path.split('.')[0])
    return '_'.join(path.split('/'))

def get_path_dict(folder):
    path_dict = dict()
    for speaker in os.listdir(folder):
        if 'txt' in speaker:
            continue
        full_speaker = cat(folder, speaker)
        for rec in os.listdir(full_speaker):
            full_rec =  cat(full_speaker, rec)
            key = make_key(cat(speaker,rec))
            path_dict[key]=full_rec.split('.')[0]

    return path_dict

def get_keys(bad_list):
    names =  []
    with open(bad_list) as f:
        names_list = f.read()

    names = [n.split(' ')[0].split('\t')[0] for n in names_list.split('\n')]
    print(f'found {len(names)} bad records')
    return names


if __name__ =='__main__':
    path_dict = get_path_dict(FOLDER)
    names =  get_keys(BAD_SANTA_LIST)
    for i, name in enumerate(names):
        if name in path_dict:
            path_to_remove =  path_dict[name]
        else:
            continue
        try:
            os.remove(path_to_remove+'.wav')
            os.remove(path_to_remove+'.txt')
            os.remove(path_to_remove+'.lab')
        except Exception as e:
            print(e)
        print(f'{i+1} Removed {path_to_remove}')
