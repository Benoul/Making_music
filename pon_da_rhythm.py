import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def midi_to_notes(filepath):
    """Extract both pitch and duration from MIDI file"""
    midi_data = pretty_midi.PrettyMIDI(filepath)
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                duration = note.end - note.start
                notes.append({
                    'pitch': note.pitch,
                    'duration': duration,
                    'velocity': note.velocity
                })
    return notes

# Sequence parameters
seq_length = 50
step = 1

midi_files = [
    "data/Queen - Bohemian Rhapsody.mid",
    "data/Queen - We Are The Champions.mid"
]

# Combine notes from all songs
all_notes = []
for filepath in midi_files:
    try:
        notes = midi_to_notes(filepath)
        all_notes.extend(notes)
        print(f"Loaded {len(notes)} notes from {filepath}")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

print(f"Total notes loaded: {len(all_notes)}")

# Quantize durations to discrete bins for easier learning
def quantize_duration(duration, bins=32):
    """Quantize duration into discrete bins"""
    duration = max(0.05, min(duration, 2.0))
    bin_idx = int((duration - 0.05) / 1.95 * (bins - 1))
    return min(bin_idx, bins - 1)

def dequantize_duration(bin_idx, bins=32):
    """Convert bin index back to duration"""
    return 0.05 + (bin_idx / (bins - 1)) * 1.95

unique_pitches = sorted(set(note['pitch'] for note in all_notes))
pitch_to_int = {pitch: i for i, pitch in enumerate(unique_pitches)}
int_to_pitch = {i: pitch for pitch, i in pitch_to_int.items()}

duration_bins = 32

encoded = []
for note in all_notes:
    pitch_token = pitch_to_int[note['pitch']]
    duration_token = quantize_duration(note['duration'], duration_bins)
    encoded.append((pitch_token, duration_token))


X_pitch, X_duration, y_pitch, y_duration = [], [], [], []
for i in range(0, len(encoded) - seq_length, step):
    seq = encoded[i:i+seq_length]
    next_note = encoded[i+seq_length]
    
    X_pitch.append([note[0] for note in seq])
    X_duration.append([note[1] for note in seq])
    y_pitch.append(next_note[0])
    y_duration.append(next_note[1])

X_pitch = torch.tensor(X_pitch)
X_duration = torch.tensor(X_duration)
y_pitch = torch.tensor(y_pitch)
y_duration = torch.tensor(y_duration)


class MusicRNN(nn.Module):
    def __init__(self, pitch_vocab_size, duration_vocab_size, 
                 embed_size=100, hidden_size=256):
        super(MusicRNN, self).__init__()

        self.pitch_embedding = nn.Embedding(pitch_vocab_size, embed_size)
        self.duration_embedding = nn.Embedding(duration_vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, batch_first=True)
        
        self.fc_pitch = nn.Linear(hidden_size, pitch_vocab_size)
        self.fc_duration = nn.Linear(hidden_size, duration_vocab_size)

    def forward(self, x_pitch, x_duration, hidden=None):

        pitch_emb = self.pitch_embedding(x_pitch)
        duration_emb = self.duration_embedding(x_duration)
        
        x = torch.cat([pitch_emb, duration_emb], dim=-1)
        
        out, hidden = self.lstm(x, hidden)
        
        pitch_out = self.fc_pitch(out)
        duration_out = self.fc_duration(out)
        
        return pitch_out, duration_out, hidden
    
model = MusicRNN(
    pitch_vocab_size=len(unique_pitches),
    duration_vocab_size=duration_bins
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_epochs = 2
for epoch in range(n_epochs):
    total_loss = 0
    for i in range(0, len(X_pitch), 16):  # batch size 16
        x_pitch_batch = X_pitch[i:i+16]
        x_duration_batch = X_duration[i:i+16]
        y_pitch_batch = y_pitch[i:i+16]
        y_duration_batch = y_duration[i:i+16]

        optimizer.zero_grad()
        
        pitch_outputs, duration_outputs, _ = model(x_pitch_batch, x_duration_batch)
        
        loss_pitch = criterion(pitch_outputs[:, -1, :], y_pitch_batch)
        loss_duration = criterion(duration_outputs[:, -1, :], y_duration_batch)
        
        loss = loss_pitch + loss_duration
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")


def generate(model, seed_pitch, seed_duration, length=100, temperature=1.0):
    """Generate music with both pitch and rhythm"""
    model.eval()
    generated_pitch = seed_pitch[:]
    generated_duration = seed_duration[:]
    
    input_pitch = torch.tensor(seed_pitch).unsqueeze(0)
    input_duration = torch.tensor(seed_duration).unsqueeze(0)
    hidden = None

    for _ in range(length):
        pitch_output, duration_output, hidden = model(input_pitch, input_duration, hidden)
        
        # Sample next pitch
        pitch_logits = pitch_output[0, -1] / temperature
        pitch_probs = F.softmax(pitch_logits, dim=0)
        next_pitch = torch.multinomial(pitch_probs, 1).item()
        
        # Sample next duration
        duration_logits = duration_output[0, -1] / temperature
        duration_probs = F.softmax(duration_logits, dim=0)
        next_duration = torch.multinomial(duration_probs, 1).item()
        
        generated_pitch.append(next_pitch)
        generated_duration.append(next_duration)
        
        # Update input sequences
        input_pitch = torch.tensor([generated_pitch[-seq_length:]]).long()
        input_duration = torch.tensor([generated_duration[-seq_length:]]).long()

    return generated_pitch, generated_duration


def notes_to_midi(pitch_sequence, duration_sequence, output_file='generated.mid'):
    """Convert pitch and duration sequences to MIDI file"""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start = 0
    
    for pitch_token, duration_token in zip(pitch_sequence, duration_sequence):
        pitch = int_to_pitch[pitch_token]
        duration = dequantize_duration(duration_token, duration_bins)
        
        note = pretty_midi.Note(
            velocity=100, 
            pitch=pitch, 
            start=start, 
            end=start + duration
        )
        instrument.notes.append(note)
        start += duration
        
    midi.instruments.append(instrument)
    midi.write(output_file)

# Generate music!!
seed_pitch = [note[0] for note in encoded[:seq_length]]
seed_duration = [note[1] for note in encoded[:seq_length]]

generated_pitch, generated_duration = generate(
    model, 
    seed_pitch=seed_pitch,
    seed_duration=seed_duration,
    length=200, 
    temperature=0.8
)

notes_to_midi(generated_pitch, generated_duration, output_file='fake_queen.mid')