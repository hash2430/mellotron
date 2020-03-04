from datasets.generate_mel_f0 import get_mel_and_f0
import matplotlib.pyplot as plt
from configs.as_is_200217 import create_hparams

hparams = create_hparams()
reference_path = 'Visualization/audio_for_pitch_contour_sample/pfp00139.wav'
gst_path = 'Visualization/audio_for_pitch_contour_sample/target-pmm_refer-pfp_topdb-25_1-gst_tts.wav'
proposed_path = 'Visualization/audio_for_pitch_contour_sample/target-pmm_refer-pfp_topdb-25_1_rel_f0_scale_min_and_max_rescale.wav'

# open wav and extract f0
_, reference_f0 = get_mel_and_f0(reference_path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)
_, gst_f0 = get_mel_and_f0(gst_path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)
_, proposed_f0 = get_mel_and_f0(proposed_path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)

reference_f0 = reference_f0.squeeze().numpy()
gst_f0 = gst_f0.squeeze().numpy()
proposed_f0 = proposed_f0.squeeze().numpy()

x_reference = range(len(reference_f0))
x_gst = range(len(gst_f0))
x_proposed = range(len(proposed_f0))

plt.plot(x_reference, reference_f0, label='Reference')
plt.plot(x_gst, gst_f0, label='GST')
plt.plot(x_proposed, proposed_f0, label='Proposed')

plt.xlabel('Frames')
plt.ylabel('F0s')

plt.title('Pitch contour comparison')
plt.legend()
plt.show()