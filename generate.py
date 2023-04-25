
import os
import glob
import time
import torch
from torch.utils.data import DataLoader

from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset
from input_representations import remi2midi
from train import D_MODEL, LEARNING_RATE, MAX_CONTEXT, MAX_STEPS, WARMUP_STEPS

MODEL = os.getenv('MODEL', 'figaro-expert-testrun')
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.getenv('ROOT_DIR', 'data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './samples')
MAX_N_FILES = int(float(os.getenv('MAX_N_FILES', -1)))
MAX_ITER = int(os.getenv('MAX_ITER', 16000))
MAX_BARS = int(os.getenv('MAX_BARS', 64))

MAKE_MEDLEYS = os.getenv('MAKE_MEDLEYS', 'False') == 'True'
N_MEDLEY_PIECES = int(os.getenv('N_MEDLEY_PIECES', 2))
N_MEDLEY_BARS = int(os.getenv('N_MEDLEY_BARS', 16))
  
CHECKPOINT = os.getenv('CHECKPOINT', 'figaro-expert')
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1))
VERBOSE = int(os.getenv('VERBOSE', 2))

def reconstruct_sample(model, batch, 
  initial_context=1, 
  output_dir=None, 
  max_iter=-1, 
  max_bars=-1,
  verbose=0,
):
  batch_size, seq_len = batch['input_ids'].shape[:2]

  batch_ = { key: batch[key][:, :initial_context] for key in ['input_ids', 'bar_ids', 'position_ids'] }
  if model.description_flavor in ['description', 'both']:
    batch_['description'] = batch['description']
    batch_['desc_bar_ids'] = batch['desc_bar_ids']
  if model.description_flavor in ['latent', 'both']:
    batch_['latents'] = batch['latents']

  max_len = seq_len + 1024
  if max_iter > 0:
    max_len = min(max_len, initial_context + max_iter)
  if verbose:
    print(f"Generating sequence ({initial_context} initial / {max_len} max length / {max_bars} max bars / {batch_size} batch size)")
  sample = model.sample(batch_, max_length=max_len, max_bars=max_bars, verbose=verbose//2)

  xs = batch['input_ids'].detach().cpu()
  xs_hat = sample['sequences'].detach().cpu()
  events = [model.vocab.decode(x) for x in xs]
  events_hat = [model.vocab.decode(x) for x in xs_hat]

  pms, pms_hat = [], []
  n_fatal = 0
  for rec, rec_hat in zip(events, events_hat):
    try:
      pm = remi2midi(rec)
      pms.append(pm)
    except Exception as err:
      print("ERROR: Could not convert events to midi:", err)
    try:
      pm_hat = remi2midi(rec_hat)
      pms_hat.append(pm_hat)
    except Exception as err:
      print("ERROR: Could not convert events to midi:", err)
      n_fatal += 1

  if output_dir:
    os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
    for pm, pm_hat, file in zip(pms, pms_hat, batch['files']):
      if verbose:
        print(f"Saving to {output_dir}/{file}")
      pm.write(os.path.join(output_dir, 'gt', file))
      pm_hat.write(os.path.join(output_dir, file))

  return events


def main():
  if MAKE_MEDLEYS:
    max_bars = N_MEDLEY_PIECES * N_MEDLEY_BARS
  else:
    max_bars = MAX_BARS

  if OUTPUT_DIR:
    params = []
    if MAKE_MEDLEYS:
      params.append(f"n_pieces={N_MEDLEY_PIECES}")
      params.append(f"n_bars={N_MEDLEY_BARS}")
    if MAX_ITER > 0:
      params.append(f"max_iter={MAX_ITER}")
    if MAX_BARS > 0:
      params.append(f"max_bars={MAX_BARS}")
    output_dir = os.path.join(OUTPUT_DIR, MODEL, ','.join(params))
  else:
    raise ValueError("OUTPUT_DIR must be specified.")

  print(f"Saving generated files to: {output_dir}")
  seq2seq_kwargs = {
    'encoder_layers': 4,
    'decoder_layers': 6,
    'num_attention_heads': 8,
    'intermediate_size': 2048,
    'd_model': D_MODEL,
    'context_size': MAX_CONTEXT,
    'lr': LEARNING_RATE,
    'warmup_steps': WARMUP_STEPS,
    'max_steps': MAX_STEPS,
  }

  model = Seq2SeqModule(
      description_flavor='description',
      **seq2seq_kwargs
    ).to(DEVICE)
  # print(model)
  if CHECKPOINT:
    ckpt = torch.load("./checkpoints/" + CHECKPOINT + ".pth")
    model.load_state_dict(ckpt)
    print('model loaded')
  model.eval()


  midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
  
  # midi_files = dm.test_ds.files
  # random.shuffle(midi_files)

  if MAX_N_FILES > 0:
    midi_files = midi_files[:MAX_N_FILES]


  description_options = None
  if MODEL in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
    description_options = model.description_options

  dataset = MidiDataset(
    ROOT_DIR,
    'test_generation',
    max_len=-1,
    description_options=description_options,
    max_bars=model.context_size,
  )


  dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate, drop_last=False)
  
  count = 0
  with torch.no_grad():
    for batch in dl:
      reconstruct_sample(model, batch, 
        output_dir=output_dir, 
        max_iter=MAX_ITER, 
        max_bars=max_bars,
        verbose=VERBOSE,
      )
      count += 1
      print('done')

if __name__ == '__main__':
  main()
