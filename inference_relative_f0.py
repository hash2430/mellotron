'''
 This inference code normalizes f0 by
 1. Subtracting reference f0 mean and adding target f0 mean to voiced frames
 2. In case - ref_f0 + target_f0 < 0, target minimum f0 is used instead.
'''

''' 
TODO: Fitting reference f0 contour into target speaker vocal range (min+alpha, max-beta) by scaling would give more natural result.
High variance from reference signal gives unnatural sounding result
'''
import sys
sys.path.append('waveglow/')

from scipy.io.wavfile import write
import librosa
import torch
from torch.utils.data import DataLoader

from configs.as_is_200217 import create_hparams
from train import load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict

hparams = create_hparams()
hparams.batch_size = 1
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)
speaker = "pmn"
checkpoint_path ='/mnt/sdc1/mellotron/as_is_200217/checkpoint_218500'
f0s_meta_path = '/mnt/sdc1/mellotron/single_init_200123/f0s_combined.txt'
    # "models/mellotron_libritts.pt"
mellotron = load_model(hparams).cuda().eval()
mellotron.load_state_dict(torch.load(checkpoint_path)['state_dict'])
waveglow_path = '/home/admin/projects/mellotron_init_with_single/models/waveglow_256channels_v4.pt'
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()
arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
audio_paths = 'data/examples_pfp.txt'
test_set = TextMelLoader(audio_paths, hparams)
datacollate = TextMelCollate(1)
dataloader = DataLoader(test_set, num_workers=1, shuffle=False,batch_size=hparams.batch_size, pin_memory=False,
                        drop_last=False, collate_fn = datacollate)
speaker_ids = TextMelLoader("filelists/wav_less_than_12s_158_speakers_train.txt", hparams).speaker_ids
speaker_id = torch.LongTensor([speaker_ids[speaker]]).cuda()

# Load mean f0
with open(f0s_meta_path, 'r', encoding='utf-8-sig') as f:
    f0s_read = f.readlines()
f0s_mean = {}
f0s_min = {}
for i in range(len(f0s_read)):
    line = f0s_read[i].split('|')
    tmp_speaker = line[0]
    f0_mean = float(line[-1])
    f0s_mean[tmp_speaker] = f0_mean
    f0_min = float(line[1])
    f0s_min[tmp_speaker] = f0_min
target_spesaker_f0 = f0s_mean[speaker]

for i, batch in enumerate(dataloader):
    reference_speaker = test_set.audiopaths_and_text[i][2]
    reference_speaker_f0 = f0s_mean[reference_speaker]
    # x: (text_padded, input_lengths, mel_padded, max_len,
    #                  output_lengths, speaker_ids, f0_padded),
    # y: (mel_padded, gate_padded)
    x, y = mellotron.parse_batch(batch)
    text_encoded = x[0]
    mel = x[2]
    pitch_contour = x[6]
    # normalize f0 for voiced frames
    mask = pitch_contour != 0.0
    pitch_contour[mask] -= reference_speaker_f0
    pitch_contour[mask] += target_spesaker_f0

    # take care of negative f0s when reference is female and target is male
    mask = pitch_contour < 0.0
    pitch_contour[mask] = f0s_min[speaker]
    tmp_nd = pitch_contour.cpu().numpy()
    tmp_mask = mask.cpu().numpy()



    with torch.no_grad():
        # get rhythm (alignment map) using tacotron 2
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.forward(x)
        rhythm = rhythm.permute(1, 0, 2)

        # Using mel as input is not generalizable. I hope there is generalizable inference method as well.
        mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference_noattention(
            (text_encoded, mel, speaker_id, pitch_contour, rhythm))

    with torch.no_grad():
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]
        audio = audio.squeeze(1).cpu().numpy()
        top_db=25
        for j in range(len(audio)):
            wav, _ = librosa.effects.trim(audio[j], top_db=top_db, frame_length=2048, hop_length=512)
            write("gen/refer_pitch_rythm_mel/{}/target-{}_refer-{}_topdb-{}_{}_rel_f0.wav".format(reference_speaker, speaker, reference_speaker, top_db, i * hparams.batch_size + j), hparams.sampling_rate, wav)
