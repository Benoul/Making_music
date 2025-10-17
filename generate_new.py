import torch.nn.functional as F
import random
import torch
import pretty_midi
from setting_up import MusicRNN, unique_notes, int_to_note, seq_length, encoded
from train import note_sequence, model

def generate(model, seed_seq, length=100, temperature=1.0):
    model.eval()
    generated = seed_seq[:]
    input_seq = torch.tensor(seed_seq).unsqueeze(0)  # (1, seq_len)
    hidden = None

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        last_logits = output[0, -1] / temperature
        probs = F.softmax(last_logits, dim=0)
        next_note = torch.multinomial(probs, 1).item()
        generated.append(next_note)
        input_seq = torch.tensor([generated[-seq_length:]]).long()

    return generated


def notes_to_midi(note_sequence, output_file='generated.mid'):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start = 0
    duration = 0.5  # or based on training data
    for note_int in note_sequence:
        pitch = int_to_note[note_int]
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start+duration)
        instrument.notes.append(note)
        start += duration
    midi.instruments.append(instrument)
    midi.write(output_file)

note_sequence = generate(model, seed_seq=encoded[:seq_length], length=200, temperature=0.8)
notes_to_midi(note_sequence, output_file='fake_queen.mid')