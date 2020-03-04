korean_public_time = [0,0]
selvas_time = [0,0]
selvas_main_time =[0]

target_meta_file_path = '/mnt/sdd1/tmp/shorter_than_12s/db_time_info.txt'
korean_files = ['/mnt/sdd1/tmp/shorter_than_12s/korean_public_shorter_tahn_12s_train.txt',
                '/mnt/sdd1/tmp/shorter_than_12s/korean_public_shorter_tahn_12s_train_sub.txt']

selvas_files = ['/mnt/sdd1/tmp/shorter_than_12s/selvas_wav_shorter_tahn_12s_train.txt',
                '/mnt/sdd1/tmp/shorter_than_12s/selvas_wav_shorter_tahn_12s_train_sub.txt']

selvas_main_files = ['/mnt/sdd1/tmp/shorter_than_12s/selvas_wav_shorter_tahn_12s_train_main.txt']

for i in range(len(korean_files)):
    with open(korean_files[i], 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for j in lines:
        korean_public_time[i] += float(j.rstrip('\n'))

for i in range(len(selvas_files)):
    with open(selvas_files[i], 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for j in lines:
        selvas_time[i] += float(j.rstrip('\n'))

for i in range(len(selvas_main_files)):
    with open(selvas_main_files[i], 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for j in lines:
        selvas_main_time[i] += float(j.rstrip('\n'))


with open(target_meta_file_path, 'w', encoding='utf-8') as f:
    f.write("Korean_public_time_first_time_included: {}h\n".format(korean_public_time[0]/3600))
    f.write("Korean_public_time_leftout_old_males included later: {}h\n".format(korean_public_time[1]/3600))
    f.write("\n")
    f.write("Selvas time first time included: {}h\n".format(selvas_time[0]/3600))
    f.write("Selvas time left out males included later: {}h\n".format(selvas_time[1]/3600))
    f.write("Selvas time Main that is included all along: {}h\n".format(selvas_main_time[0]/3600))
    f.write("\nSummirization\n")
    f.write("Korean public DB total: {}h\n".format((korean_public_time[0]+korean_public_time[1])/3600))
    f.write("Selvas DB total: {}h\n".format((selvas_time[0]+selvas_time[1])/3600))
    f.write("Selvas multispeaker only: {}h\n".format((selvas_time[0]+selvas_time[1]-selvas_main_time[0])/3600))
