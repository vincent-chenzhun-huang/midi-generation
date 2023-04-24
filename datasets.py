import random
import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm

from input_representations import InputRepresentation
from vocab import RemiVocab, DescriptionVocab
from constants import (
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY,
    TIME_SIGNATURE_KEY, INSTRUMENT_KEY, CHORD_KEY,
    NOTE_DENSITY_KEY, MEAN_PITCH_KEY, MEAN_VELOCITY_KEY, MEAN_DURATION_KEY
)


CACHE_PATH = os.getenv('CACHE_PATH', os.getenv(
    'SCRATCH', os.getenv('TMPDIR', './temp')))
LATENT_CACHE_PATH = os.getenv('LATENT_CACHE_PATH', os.path.join(
    os.getenv('SCRATCH', os.getenv('TMPDIR', './temp')), 'latent'))

class MidiDataset(IterableDataset):
    def __init__(self,
                 root,
                 partition,
                 max_len,
                 description_options=None,
                 group_bars=False,
                 max_bars=512,
                 max_positions=512,
                 max_bars_per_context=-1,
                 max_contexts_per_file=-1,
                 bar_token_mask=None,
                 bar_token_idx=2,
                 pad_token=0,
                 context_size=512,
                 shuffle=False,
                 debug=False):
        self.files = sorted([os.path.join(root, partition, f) for f in os.listdir(os.path.join(root, partition))])
        if shuffle:
            random.shuffle(self.files)
        self.group_bars = group_bars
        self.max_len = max_len
        self.max_bars = max_bars
        self.max_positions = max_positions
        self.max_bars_per_context = max_bars_per_context
        self.max_contexts_per_file = max_contexts_per_file
        self.debug = debug

        self.vocab = RemiVocab()

        self.description_options = description_options

        self.desc_vocab = DescriptionVocab()

        self.bar_token_mask = bar_token_mask
        self.bar_token_idx = bar_token_idx
        self.pad_token = pad_token
        self.context_size = context_size

    def __iter__(self):
        for i in tqdm(range(len(self.files))):
            try:
                current_file = self.load_file(self.files[i])
            except ValueError as err:
                if self.debug:
                    print(err)
                continue

            events = current_file['events']

            # Identify start of bars
            bars, bar_ids = self.get_bars(events, include_ids=True)
            if len(bars) > self.max_bars:
                if self.debug:
                    print(
                        f"WARNING: REMI sequence has more than {self.max_bars} bars: {len(bars)} event bars.")
                continue

            # Identify positions
            position_ids = self.get_positions(events)
            max_pos = position_ids.max()
            if max_pos > self.max_positions:
                if self.debug:
                    print(
                        f"WARNING: REMI sequence has more than {self.max_positions} positions: {max_pos.item()} positions found")
                continue

            # Mask bar tokens if required
            if self.bar_token_mask is not None and self.max_bars_per_context > 0:
                events = self.mask_bar_tokens(
                    events, bar_token_mask=self.bar_token_mask)

            # Encode tokens with appropriate vocabulary
            event_ids = torch.tensor(
                self.vocab.encode(events), dtype=torch.long)

            bos, eos = self.get_bos_eos_events()
            zero = torch.tensor([0], dtype=torch.int)

            if self.max_bars_per_context and self.max_bars_per_context > 0:
                # Find all indices where a new context starts based on number of bars per context
                starts = [bars[i]
                          for i in range(0, len(bars), self.max_bars_per_context)]
                # Convert starts to ranges
                contexts = list(
                    zip(starts[:-1], starts[1:])) + [(starts[-1], len(event_ids))]
                # # Limit the size of the range if it's larger than the max. context size
                # contexts = [(max(start, end - (self.max_len+1)), end) for (start, end) in contexts]

            else:
                event_ids = torch.cat([bos, event_ids, eos])
                bar_ids = torch.cat([zero, bar_ids, zero])
                position_ids = torch.cat([zero, position_ids, zero])

                if self.max_len > 0:
                    starts = list(range(0, len(event_ids), self.max_len+1))
                    if len(starts) > 1:
                        contexts = [(start, start + self.max_len+1) for start in starts[:-1]] + [
                            (len(event_ids) - (self.max_len+1), len(event_ids))]
                    elif len(starts) > 0:
                        contexts = [(starts[0], self.max_len+1)]
                else:
                    contexts = [(0, len(event_ids))]

            if self.max_contexts_per_file and self.max_contexts_per_file > 0:
                contexts = contexts[:self.max_contexts_per_file]

            for start, end in contexts:
                # Add <bos> and <eos> to each context if contexts are limited to a certain number of bars
                if self.max_bars_per_context and self.max_bars_per_context > 0:
                    src = torch.cat([bos, event_ids[start:end], eos])
                    b_ids = torch.cat([zero, bar_ids[start:end], zero])
                    p_ids = torch.cat([zero, position_ids[start:end], zero])
                else:
                    src = event_ids[start:end]
                    b_ids = bar_ids[start:end]
                    p_ids = position_ids[start:end]

                if self.max_len > 0:
                    src = src[:self.max_len + 1]

                x = {
                    'input_ids': src,
                    'file': os.path.basename(self.files[i]),
                    'bar_ids': b_ids,
                    'position_ids': p_ids,
                }

                # Assume that bar_ids are in ascending order (except for EOS)
                min_bar = b_ids[0]
                desc_events = current_file['description']
                desc_bars = [i for i, event in enumerate(
                    desc_events) if f"{BAR_KEY}_" in event]
                # subtract one since first bar has id == 1
                start_idx = desc_bars[max(0, min_bar - 1)]

                desc_bar_ids = torch.zeros(
                    len(desc_events), dtype=torch.int)
                desc_bar_ids[desc_bars] = 1
                desc_bar_ids = torch.cumsum(desc_bar_ids, dim=0)

                if self.max_bars_per_context and self.max_bars_per_context > 0:
                    end_idx = desc_bars[min_bar +
                                        self.max_bars_per_context]
                    desc_events = desc_events[start_idx:end_idx]
                    desc_bar_ids = desc_bar_ids[start_idx:end_idx]
                    start_idx = 0

                desc_bos = torch.tensor(
                    self.desc_vocab.encode([BOS_TOKEN]), dtype=torch.int)
                desc_eos = torch.tensor(
                    self.desc_vocab.encode([EOS_TOKEN]), dtype=torch.int)
                desc_ids = torch.tensor(
                    self.desc_vocab.encode(desc_events), dtype=torch.int)
                if min_bar == 0:
                    desc_ids = torch.cat([desc_bos, desc_ids, desc_eos])
                    desc_bar_ids = torch.cat([zero, desc_bar_ids, zero])
                else:
                    desc_ids = torch.cat([desc_ids, desc_eos])
                    desc_bar_ids = torch.cat([desc_bar_ids, zero])

                if self.max_len > 0:
                    start, end = start_idx, start_idx + self.max_len + 1
                    x['description'] = desc_ids[start:end]
                    x['desc_bar_ids'] = desc_bar_ids[start:end]
                else:
                    x['description'] = desc_ids[start:]
                    x['desc_bar_ids'] = desc_bar_ids[start:]

                yield x

    def get_bars(self, events, include_ids=False):
        bars = [i for i, event in enumerate(events) if f"{BAR_KEY}_" in event]

        if include_ids:
            bar_ids = torch.bincount(torch.tensor(
                bars, dtype=torch.int), minlength=len(events))
            bar_ids = torch.cumsum(bar_ids, dim=0)

            return bars, bar_ids
        else:
            return bars

    def get_positions(self, events):
        events = [
            f"{POSITION_KEY}_0" if f"{BAR_KEY}_" in event else event for event in events]
        position_events = [
            event if f"{POSITION_KEY}_" in event else None for event in events]

        positions = [int(pos.split('_')[-1])
                     if pos is not None else None for pos in position_events]

        if positions[0] is None:
            positions[0] = 0
        for i in range(1, len(positions)):
            if positions[i] is None:
                positions[i] = positions[i-1]
        positions = torch.tensor(positions, dtype=torch.int)

        return positions

    def mask_bar_tokens(self, events, bar_token_mask='<mask>'):
        events = [
            bar_token_mask if f'{BAR_KEY}_' in token else token for token in events]
        return events

    def get_bos_eos_events(self, tuple_size=8):
        bos_event = torch.tensor(
            self.vocab.encode([BOS_TOKEN]), dtype=torch.long)
        eos_event = torch.tensor(
            self.vocab.encode([EOS_TOKEN]), dtype=torch.long)
        return bos_event, eos_event

    def preprocess_description(self, desc, instruments=True, chords=True, meta=True):
        valid_keys = {
            BAR_KEY: True,
            INSTRUMENT_KEY: instruments,
            CHORD_KEY: chords,
            TIME_SIGNATURE_KEY: meta,
            NOTE_DENSITY_KEY: meta,
            MEAN_PITCH_KEY: meta,
            MEAN_VELOCITY_KEY: meta,
            MEAN_DURATION_KEY: meta,
        }
        return [token for token in desc if len(token.split('_')) == 0 or valid_keys[token.split('_')[0]]]

    def load_file(self, file):
        sample = {}
        try:
            rep = InputRepresentation(file, strict=True)
            events = rep.get_remi_events()
            description = rep.get_description()
        except Exception as err:
            raise ValueError(f'Unable to load file {file}') from err

        sample = {
            'events': events,
            'description': description
        }
        return sample
    
    def collate(self, features):
        batch = {}

        xs = [feature['input_ids'] for feature in features]
        xs = pad_sequence(xs, batch_first=True, padding_value=self.pad_token)

        if self.context_size > 0:
            max_len = self.context_size
            max_desc_len = self.context_size
        else:
            max_len = xs.size(1)
            max_desc_len = int(1e4)

        tmp = xs[:, :(max_len + 1)][:, :-1]
        labels = xs[:, :(max_len + 1)][:, 1:].clone().detach()
        xs = tmp

        seq_len = xs.size(1)

        batch['input_ids'] = xs
        batch['labels'] = labels

        if 'position_ids' in features[0]:
            position_ids = [feature['position_ids'] for feature in features]
            position_ids = pad_sequence(
                position_ids, batch_first=True, padding_value=0)
            batch['position_ids'] = position_ids[:, :seq_len]

        if 'bar_ids' in features[0]:
            bar_ids = [feature['bar_ids'] for feature in features]
            bar_ids = pad_sequence(bar_ids, batch_first=True, padding_value=0)
            batch['bar_ids'] = bar_ids[:, :seq_len]

        if 'codes' in features[0]:
            codes = [feature['codes'] for feature in features]
            codes = pad_sequence(codes, batch_first=True, padding_value=0)
            batch['codes'] = codes[:, :max_desc_len]

        if 'description' in features[0]:
            description = [feature['description'] for feature in features]
            description = pad_sequence(
                description, batch_first=True, padding_value=self.pad_token)
            desc = description[:, :max_desc_len]
            batch['description'] = desc

            if 'desc_bar_ids' in features[0]:
                desc_len = desc.size(1)
                desc_bar_ids = [feature['desc_bar_ids']
                                for feature in features]
                desc_bar_ids = pad_sequence(
                    desc_bar_ids, batch_first=True, padding_value=0)
                batch['desc_bar_ids'] = desc_bar_ids[:, :desc_len]

        if 'file' in features[0]:
            batch['files'] = [feature['file'] for feature in features]
        return batch
    

if __name__ == '__main__':
  dataset = MidiDataset('./data', 'toy_data', 10, shuffle=False)
  
  for i, sample in enumerate(dataset):
    print(sample)
    if i >= 3:
      break