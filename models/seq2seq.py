import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from datasets import MidiDataset
from vocab import RemiVocab, DescriptionVocab
from constants import PAD_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY


import transformers
from transformers import (
  BertConfig,
  EncoderDecoderConfig,
  EncoderDecoderModel
)

class GroupEmbedding(nn.Module):
  def __init__(self, n_tokens, n_groups, out_dim, inner_dim=128):
    super().__init__()
    self.n_tokens = n_tokens
    self.n_groups = n_groups
    self.inner_dim = inner_dim
    self.out_dim = out_dim

    self.embedding = nn.Embedding(n_tokens, inner_dim)
    self.proj = nn.Linear(n_groups * inner_dim, out_dim, bias=False)

  def forward(self, x):
    shape = x.shape
    emb = self.embedding(x)
    return self.proj(emb.view(*shape[:-1], self.n_groups * self.inner_dim))

class Seq2SeqModule(torch.nn.Module):
  def __init__(self,
               d_model=512,
               d_latent=512,
               n_codes=512,
               n_groups=8,
               context_size=512,
               lr=1e-4,
               lr_schedule='sqrt_decay',
               warmup_steps=None,
               max_steps=None,
               encoder_layers=6,
               decoder_layers=12,
               intermediate_size=2048,
               num_attention_heads=8,
               description_flavor='description',
               description_options=None,
               use_pretrained_latent_embeddings=True):
    super(Seq2SeqModule, self).__init__()
    self.description_flavor = description_flavor
    assert self.description_flavor in ['description'], f"Unknown description flavor '{self.description_flavor}', expected one of ['description']"
    self.description_options = description_options

    self.context_size = context_size
    self.d_model = d_model
    self.d_latent = d_latent

    self.lr = lr
    self.lr_schedule = lr_schedule
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps

    self.vocab = RemiVocab()

    encoder_config = BertConfig(
      vocab_size=1,
      pad_token_id=0,
      hidden_size=self.d_model,
      num_hidden_layers=encoder_layers,
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      max_position_embeddings=1024,
      position_embedding_type='relative_key_query'
    )
    decoder_config = BertConfig(
      vocab_size=1,
      pad_token_id=0,
      hidden_size=self.d_model,
      num_hidden_layers=decoder_layers,
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      max_position_embeddings=1024,
      position_embedding_type='relative_key_query'
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    self.transformer = EncoderDecoderModel(config)
    self.transformer.config.decoder.is_decoder = True
    self.transformer.config.decoder.add_cross_attention = True


    self.max_bars = self.context_size
    self.max_positions = 512
    self.bar_embedding = nn.Embedding(self.max_bars + 1, self.d_model)
    self.pos_embedding = nn.Embedding(self.max_positions + 1, self.d_model)

    desc_vocab = DescriptionVocab()
    self.desc_in = nn.Embedding(len(desc_vocab), self.d_model)

    self.in_layer = nn.Embedding(len(self.vocab), self.d_model)
    self.out_layer = nn.Linear(self.d_model, len(self.vocab), bias=False)
    
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.to_i(PAD_TOKEN))
        
    # self.save_hyperparameters()

  def encode(self, z, desc_bar_ids=None):
    z_emb = self.desc_in(z)
    if desc_bar_ids is not None:
      z_emb += self.bar_embedding(desc_bar_ids)

    out = self.transformer.encoder(inputs_embeds=z_emb, output_hidden_states=True)
    encoder_hidden = out.hidden_states[-1]
    return encoder_hidden

  def decode(self, x, labels=None, bar_ids=None, position_ids=None, encoder_hidden_states=None, return_hidden=False):
    seq_len = x.size(1)

    # Shape of x_emb: (batch_size, seq_len, d_model)
    x_emb = self.in_layer(x)
    if bar_ids is not None:
      x_emb += self.bar_embedding(bar_ids)
    if position_ids is not None:
      x_emb += self.pos_embedding(position_ids)

    # # Add latent embedding to input embeddings
    # if bar_ids is not None:
    #   assert bar_ids.max() <= encoder_hidden.size(1)
    #   embs = torch.cat([torch.zeros(x.size(0), 1, self.d_model, device=self.device), encoder_hidden], dim=1)
    #   offset = (seq_len * torch.arange(bar_ids.size(0), device=self.device)).unsqueeze(1)
    #   # Use bar_ids to gather encoder hidden states s.t. latent_emb[i, j] == encoder_hidden[i, bar_ids[i, j]]
    #   latent_emb = F.embedding((bar_ids + offset).view(-1), embs.view(-1, self.d_model)).view(x_emb.shape)
    #   x_emb += latent_emb

    if encoder_hidden_states is not None:
      # Make x_emb and encoder_hidden_states match in sequence length. Necessary for relative positional embeddings
      padded = pad_sequence([x_emb.transpose(0, 1), encoder_hidden_states.transpose(0, 1)], batch_first=True)
      x_emb, encoder_hidden_states = padded.transpose(1, 2)

      out = self.transformer.decoder(
        inputs_embeds=x_emb, 
        encoder_hidden_states=encoder_hidden_states, 
        output_hidden_states=True
      )
      hidden = out.hidden_states[-1][:, :seq_len]
    else:
      out = self.transformer.decoder(inputs_embeds=x_emb, output_hidden_states=True)
      hidden = out.hidden_states[-1][:, :seq_len]

    # Shape of logits: (batch_size, seq_len, tuple_size, vocab_size)

    if return_hidden:
      return hidden
    else:
      return self.out_layer(hidden)


  def forward(self, x, z=None, labels=None, position_ids=None, bar_ids=None, description_bar_ids=None, return_hidden=False):
    encoder_hidden = self.encode(z, desc_bar_ids=description_bar_ids)

    out = self.decode(x, 
      labels=labels, 
      bar_ids=bar_ids, 
      position_ids=position_ids, 
      encoder_hidden_states=encoder_hidden,
      return_hidden=return_hidden
    )

    return out 
    
  def get_loss(self, batch, return_logits=False):
    # Shape of x: (batch_size, seq_len, tuple_size)
    x = batch['input_ids']
    bar_ids = batch['bar_ids']
    position_ids = batch['position_ids']
    # Shape of labels: (batch_size, tgt_len, tuple_size)
    labels = batch['labels']

    # Shape of z: (batch_size, context_size, n_groups, d_latent)
    z = batch['description']
    desc_bar_ids = batch['desc_bar_ids']

    logits = self(x, z=z, labels=labels, bar_ids=bar_ids, position_ids=position_ids, description_bar_ids=desc_bar_ids)
    # Shape of logits: (batch_size, tgt_len, tuple_size, vocab_size)
    pred = logits.view(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    
    loss = self.loss_fn(pred, labels)

    if return_logits:
      return loss, logits
    else:
      return loss