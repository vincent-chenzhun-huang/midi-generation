import mido
from mido import MidiFile, MidiTrack
import pretty_midi

def transpose_midi_file(input_file, output_file, semitones):
    # Read the input MIDI file
    midi = MidiFile(input_file)

    # Iterate through the tracks and messages in the MIDI file
    for track in midi.tracks:
        for msg in track:
            # Check if the message is a 'note_on' or 'note_off' message
            if msg.type in ['note_on', 'note_off']:
                # Transpose the note by the given number of semitones
                msg.note += semitones
                # Clamp the note value to the valid MIDI range (0-127)
                msg.note = max(0, min(msg.note, 127))

    # Save the transposed MIDI file
    midi.save(output_file)
    
def remove_drum_track(input_file, output_file):
    # Read the input MIDI file
    midi = MidiFile(input_file)

    # Create a new MIDI file for the output
    output_midi = MidiFile()

    # Print track information
    for i, track in enumerate(midi.tracks):
        print(f'Track {i}: {track.name}')

    # Loop through the input MIDI tracks
    for track in midi.tracks:
        new_track = MidiTrack()
        output_midi.tracks.append(new_track)

        # Loop through the messages in the input track
        for msg in track:
            # Check if the message is not a drum track message (channel != 9)
            if not hasattr(msg, 'channel') or msg.channel != 9:
                new_track.append(msg)

    # Save the modified MIDI file
    output_midi.save(output_file)



input_file = 'AutumnLeaves-original.mid'
intermediate_file = 'autumn_leaves_no_drum.mid'
out_file = 'autumn_leaves_no_drum.mid'
semitones = 4  # Transpose up by 5 semitones

remove_drum_track(input_file, intermediate_file)