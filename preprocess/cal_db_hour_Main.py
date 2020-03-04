import os
file_list = '/home/admin/projects/mellotron_as_is/filelists/wav_less_than_12s_158_speakers_train.txt'
return_path = '/mnt/sdd1/tmp/selvas_wav_shorter_tahn_12s_train_main.txt'
with open(file_list, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
paths =[]
for line in lines:
    path = line.split('|')[0]
    paths.append(path)

for path in paths:
    if path.startswith('/mnt/sdd1/selvas_wav/Main'):
        command = 'soxi -D {} >> {}'.format(path, return_path)
        os.system(command)

