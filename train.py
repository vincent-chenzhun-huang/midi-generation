
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import math
from datasets import MidiDataset
from vocab import RemiVocab, DescriptionVocab
from constants import PAD_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY
from itertools import cycle, islice
from tqdm import tqdm
import pytorch_lightning as pl

import transformers
from transformers import (
  BertConfig,
  EncoderDecoderConfig,
  EncoderDecoderModel
)

import os
import glob

import wandb

from models.seq2seq import Seq2SeqModule
# from models.seq2seq_org import Seq2SeqModule as Seq2SeqModuleOrg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.getenv('ROOT_DIR', './lmd_full')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './results')
LOGGING_DIR = os.getenv('LOGGING_DIR', './logs')
MAX_N_FILES = int(os.getenv('MAX_N_FILES', -1))

MODEL = os.getenv('MODEL', None)
MODEL_NAME = os.getenv('MODEL_NAME', None)
N_CODES = int(os.getenv('N_CODES', 2048))
N_GROUPS = int(os.getenv('N_GROUPS', 16))
D_MODEL = int(os.getenv('D_MODEL', 512))
D_LATENT = int(os.getenv('D_LATENT', 1024))

CHECKPOINT = os.getenv('CHECKPOINT', None)
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
TARGET_BATCH_SIZE = int(os.getenv('TARGET_BATCH_SIZE', 512))

EPOCHS = int(os.getenv('EPOCHS', '16'))
WARMUP_STEPS = int(float(os.getenv('WARMUP_STEPS', 4000)))
MAX_STEPS = int(float(os.getenv('MAX_STEPS', 1e20)))
MAX_TRAINING_STEPS = int(float(os.getenv('MAX_TRAINING_STEPS', 100000)))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
LR_SCHEDULE = os.getenv('LR_SCHEDULE', 'const')
CONTEXT_SIZE = int(os.getenv('CONTEXT_SIZE', 256))

ACCUMULATE_GRADS = max(1, TARGET_BATCH_SIZE//BATCH_SIZE)

N_WORKERS = min(os.cpu_count(), float(os.getenv('N_WORKERS', 'inf')))
if device.type == 'cuda':
  N_WORKERS = min(N_WORKERS, 8*torch.cuda.device_count())
N_WORKERS = int(N_WORKERS)
MAX_CONTEXT = min(1024, CONTEXT_SIZE)

def configure_optimizers(model, lr=1, schedule='sqrt_decay'):
  # set LR to 1, scale with LambdaLR scheduler
  optimizer = transformers.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

  if schedule == 'sqrt_decay':
    # constant warmup, then 1/sqrt(n) decay starting from the initial LR
    lr_func = lambda step: min(lr, lr / math.sqrt(max(step, 1)/WARMUP_STEPS))
  elif schedule == 'linear':
    # linear warmup, linear decay
    lr_func = lambda step: min(LEARNING_RATE, LEARNING_RATE*step/WARMUP_STEPS, LEARNING_RATE*(1 - (step - WARMUP_STEPS)/MAX_STEPS))
  elif schedule == 'cosine':
    # linear warmup, cosine decay to 10% of initial LR
    lr_func = lambda step: LEARNING_RATE * min(step/WARMUP_STEPS, 0.55 + 0.45*math.cos(math.pi*(min(step, MAX_STEPS) - WARMUP_STEPS)/(MAX_STEPS - WARMUP_STEPS)))
  else:
    # Use no lr scheduling
    lr_func = lambda step: LEARNING_RATE
  
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
  return optimizer, scheduler

def load_data():
  #create dataset
    train_ds = MidiDataset('./data', 'train', MAX_CONTEXT, shuffle=True)
    valid_ds = MidiDataset('./data', 'val', MAX_CONTEXT, shuffle=False)
    test_ds = MidiDataset('./data', 'test', MAX_CONTEXT, shuffle=False)

    train_loader = DataLoader(train_ds, 
                      collate_fn=train_ds.collate, 
                      batch_size=BATCH_SIZE,
                      pin_memory=True, 
                      num_workers=4)
    val_loader = DataLoader(valid_ds, 
                      collate_fn=valid_ds.collate, 
                      batch_size=BATCH_SIZE, 
                      pin_memory=True, 
                      num_workers=4)
    test_loader = DataLoader(test_ds, 
                      collate_fn=test_ds.collate, 
                      batch_size=BATCH_SIZE, 
                      pin_memory=True, 
                      num_workers=4)
    return train_loader, val_loader, test_loader

def train(model, train_loader, optimizer, loss_fn):
  model.train()
  total_loss = 0
  count = 0
  for i, batch in tqdm(enumerate(train_loader)):
    optimizer.zero_grad()
    x = batch['input_ids'].to(device)
    z = batch['description'].to(device)
    labels = batch['labels'].to(device)
    position_ids = batch['position_ids'].to(device)
    bar_ids = batch['bar_ids'].to(device)
    description_bar_ids = batch['desc_bar_ids'].to(device)

    logits = model.forward(x, z=z, labels=labels, 
                        position_ids=position_ids, bar_ids=bar_ids, 
                        description_bar_ids=description_bar_ids, 
                        return_hidden=False)
    pred = logits.view(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    loss = loss_fn(pred, labels)
    total_loss += loss.item()

    loss.backward()
    optimizer.step()
    count += 1

    del x, z, labels, position_ids, bar_ids, description_bar_ids, logits, pred, loss
    torch.cuda.empty_cache()
  return total_loss / count

def validate(model, val_loader, loss_fn):
  model.eval()
  total_loss = 0
  count = 0
  with torch.no_grad():
    for i, batch in enumerate(val_loader):
      x = batch['input_ids'].to(device)
      z = batch['description'].to(device)
      labels = batch['labels'].to(device)
      position_ids = batch['position_ids'].to(device)
      bar_ids = batch['bar_ids'].to(device)
      description_bar_ids = batch['desc_bar_ids'].to(device)

      logits = model.forward(x, z=z, labels=labels, 
                          position_ids=position_ids, bar_ids=bar_ids, 
                          description_bar_ids=description_bar_ids, 
                          return_hidden=False)
      pred = logits.view(-1, logits.shape[-1])
      labels = labels.reshape(-1)
      loss = model.loss_fn(pred, labels)
      total_loss += loss.item()

      y = batch['labels']
      pad_token_id = model.vocab.to_i(PAD_TOKEN)
      
      logits = logits.view(logits.size(0), -1, logits.size(-1))
      y = y.view(y.size(0), -1).to(device)

      log_pr = logits.log_softmax(dim=-1).to(device)
      log_pr[y == pad_token_id] = 0 # log(pr) = log(1) for padding
      log_pr = torch.gather(log_pr, -1, y.unsqueeze(-1)).squeeze(-1)

      t = (y != pad_token_id).sum(dim=-1)
      ppl = (-log_pr.sum(dim=1) / t).exp().mean()
      
      count += 1

      del x, z, labels, position_ids, bar_ids, description_bar_ids, logits, pred, loss
      torch.cuda.empty_cache()
  return total_loss / count, ppl

def train_val_loop(model):
  print('starting training')
  train_loader, val_loader, test_loader = load_data()
  print('loaded data')
  optimizer, scheduler = configure_optimizers(model, lr=LEARNING_RATE, schedule=LR_SCHEDULE)
  print('configured optimizers')
  loss_fn = nn.CrossEntropyLoss(ignore_index=model.vocab.to_i(PAD_TOKEN))
  best_val_loss = float('inf')
  best_val_ppl = float('inf')
  best_epoch = 0

  run = wandb.init(
    name = "expert", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # id = '2qd3d0vl', ### Insert specific run id here if you want to resume a previous run
    # resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "artsml", ### Project should be created in your wandb account 
    config = {} ### Wandb Config for your run
  )

  for epoch in range(1, EPOCHS+1):
    print(f'epoch {epoch}')
    train_loss = train(model, train_loader, optimizer, loss_fn)
    val_loss, val_ppl = validate(model, val_loader, loss_fn)
    curr_lr = float(optimizer.param_groups[0]['lr'])
    wandb.log({
        'train_loss': train_loss,  
        'val_loss': val_loss, 
        'lr'        : curr_lr
    })

    print(f'train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, val ppl: {val_ppl:.3f}')
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_val_ppl = val_ppl
      best_epoch = epoch
      torch.save(
          {'model_state_dict'         : model.state_dict(),
          'optimizer_state_dict'     : optimizer.state_dict(),
          'scheduler_state_dict'     : scheduler.state_dict(),
          'val_loss'                  : val_loss, 
          'epoch'                    : epoch}, 
          f'./checkpoints/{MODEL}-best.ckpt'
      )
    scheduler.step()

TRAINING_STEPS = 0

def main():
  ### Define available models ###

  available_models = ['figaro-expert']

  assert MODEL is not None, 'the MODEL needs to be specified'
  # assert MODEL in available_models, f"unknown MODEL: {MODEL}"
  vae_module = None

  ### Create and train model ###

  # load model from checkpoint if available
  # wandb.login(key="99caa13ec9552adf0e92e5c30021307ce3cf7fa4")
    # model_class = {
    #   'figaro-expert': Seq2SeqModule,
    # }[MODEL]
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

    # use lambda functions for lazy initialization
  model = Seq2SeqModule(
      description_flavor='description',
      **seq2seq_kwargs
    ).to(device)
  # print(model)
  if CHECKPOINT:
    # ckpt = Seq2SeqModuleOrg.load_from_checkpoint("./checkpoints/" + CHECKPOINT + ".ckpt")
    # torch.save(ckpt.state_dict(), "./checkpoints/" + CHECKPOINT + ".sd")
    ckpt = torch.load("./checkpoints/" + CHECKPOINT + ".ckpt")
    # print(ckpt.keys())
    model.load_state_dict(ckpt['state_dict'])
    # checkpoint = torch.load("./checkpoints/" + CHECKPOINT + ".ckpt")
    # print(checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = torch.load()['model_state_dict']
  train_val_loop(model)

if __name__ == '__main__':
  main()