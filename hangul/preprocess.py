import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import blizzard, ljspeech, sitec, selvas
from hparams import hparams


def preprocess_blizzard(args):
  in_dir = os.path.join(args.base_dir, 'Blizzard2012')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = blizzard.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_ljspeech(args):
  in_dir = os.path.join(args.base_dir, 'LJSpeech-1.0')
  # out_dir = os.path.join(args.base_dir, args.output)
  # os.makedirs(out_dir, exist_ok=True)
  ljspeech.build_from_path(in_dir, args.base_dir, args.output, args.num_workers, tqdm=tqdm)
  # metadata = ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  # write_metadata(metadata, out_dir)


def preprocess_sitec(args):
  in_dir = os.path.join(args.base_dir, 'sitec')
  # out_dir = os.path.join(args.base_dir, args.output)
  # os.makedirs(out_dir, exist_ok=True)
  # metadata = sitec.build_from_path(in_dir, args.base_dir, args.output, args.num_workers, tqdm=tqdm)
  sitec.build_from_path(in_dir, args.base_dir, args.output, args.num_workers, tqdm=tqdm)
  # metadata = sitec.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  # write_metadata(metadata, out_dir)


def preprocess_sitec_short(args):
  in_dir = os.path.join(args.base_dir, 'sitec_short')
  out_dir = os.path.join(args.base_dir, 'training_'+args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = sitec.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)

def preprocess_selvas_long(args):
  in_dir = '/SpeechDB/female/01.Main'  # '/home/yj/emotiontts/SpeechDB/female/01.Main'
  # out_dir = os.path.join(args.base_dir, args.output)
  # os.makedirs(out_dir, exist_ok=True)
  # metadata = sitec.build_from_path(in_dir, args.base_dir, args.output, args.num_workers, tqdm=tqdm)
  selvas.build_from_path(in_dir, args.base_dir, args.output, args.num_workers, tqdm=tqdm)
  # metadata = sitec.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  # write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[2] for m in metadata])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max(m[2] for m in metadata))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/work/yeonjoo/tacotron'))
  parser.add_argument('--output', default='sitec')
  parser.add_argument('--dataset', required=True, choices=['blizzard', 'ljspeech', 'sitec', 'sitec_short', 'selvas_long'])
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  args = parser.parse_args()
  hparams.parse(args.hparams)
  if args.dataset == 'blizzard':
    preprocess_blizzard(args)
  elif args.dataset == 'ljspeech':
    preprocess_ljspeech(args)
  elif args.dataset == 'sitec':
    preprocess_sitec(args)
  elif args.dataset == 'sitec_short':
    preprocess_sitec_short(args)
  elif args.dataset == 'selvas_long':
    preprocess_selvas_long(args)


if __name__ == "__main__":
  main()
