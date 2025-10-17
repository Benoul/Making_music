import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def midi_to_notes(filepath):
    midi_data = pretty_midi.PrettyMIDI(filepath)
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append(note.pitch)  # MIDI pitch (0–127)
    return sorted(notes)

# Let's say we use sequences of 50 notes to predict the next note
seq_length = 50
step = 1

all_notes = midi_to_notes("data/Queen - Bohemian Rhapsody.mid")

# Encode note pitches to integer tokens
unique_notes = sorted(set(all_notes))
note_to_int = {note: i for i, note in enumerate(unique_notes)}
int_to_note = {i: note for note, i in note_to_int.items()}

# Convert notes to integers
encoded = [note_to_int[n] for n in all_notes]

# Create input/output pairs
X, y = [], []
for i in range(0, len(encoded) - seq_length, step):
    X.append(encoded[i:i+seq_length])
    y.append(encoded[i+seq_length])

X = torch.tensor(X)
y = torch.tensor(y)


class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=256):
        super(MusicRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
    
model = MusicRNN(vocab_size=len(unique_notes))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_epochs = 2
for epoch in range(n_epochs):
    total_loss = 0
    for i in range(0, len(X), 16):  # batch size 32
        x_batch = X[i:i+16]
        y_batch = y[i:i+16]

        optimizer.zero_grad()
        outputs, _ = model(x_batch)
        loss = criterion(outputs[:, -1, :], y_batch)  # last output only
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")


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