import torch
import pytorch_lightning as pl

# a collection of utility functions that converts MIDI to "expert descriptions"
def convert_lightning_checkpoint(ckpt_file, output_file):
    # Load the pytorch_lightning checkpoint
    lightning_checkpoint = torch.load(ckpt_file, map_location='cpu')

    # Extract the state_dict from the checkpoint
    state_dict = lightning_checkpoint['state_dict']

    # Remove "model." prefix from the state_dict keys (if present)
    converted_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "", 1)  # Remove the "model." prefix
        converted_state_dict[new_key] = value

    # Save the converted state_dict
    torch.save(converted_state_dict, output_file)

ckpt_file = '/home/vincent-huang/cmu/midi-generation/checkpoints/figaro-expert.ckpt'
output_file = '/home/vincent-huang/cmu/midi-generation/checkpoints/figaro-expert.pth'

convert_lightning_checkpoint(ckpt_file, output_file)