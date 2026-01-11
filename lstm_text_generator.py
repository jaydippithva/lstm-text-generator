import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import string



with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

# Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

print("Total characters:", len(text))

# -----------------------------
# 2. TOKENIZATION (Character-level)
# -----------------------------
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)
print("Vocabulary size:", vocab_size)

# Convert text to numbers
encoded_text = np.array([char_to_idx[c] for c in text])

# -----------------------------
# 3. CREATE INPUT-OUTPUT SEQUENCES
# -----------------------------
seq_length = 100
inputs = []
targets = []

for i in range(0, len(encoded_text) - seq_length):
    inputs.append(encoded_text[i:i + seq_length])
    targets.append(encoded_text[i + seq_length])

X = np.array(inputs)
y = np.array(targets)

print("Input shape:", X.shape)
print("Target shape:", y.shape)

# -----------------------------
# 4. BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=seq_length),
    LSTM(128, return_sequences=False),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam"
)

model.summary()

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
early_stop = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X, y,
    batch_size=128,
    epochs=20,
    callbacks=[early_stop]
)


def generate_text(seed_text, length=500):
    seed_text = seed_text.lower()
    generated = seed_text

    for _ in range(length):
        encoded_seed = [char_to_idx.get(c, 0) for c in generated[-seq_length:]]
        encoded_seed = np.pad(encoded_seed,
                              (seq_length - len(encoded_seed), 0))
        encoded_seed = np.reshape(encoded_seed, (1, seq_length))

        prediction = model.predict(encoded_seed, verbose=0)
        next_char = idx_to_char[np.argmax(prediction)]

        generated += next_char

    return generated


seed = "to be or not to be"
output = generate_text(seed)
print("\nGenerated Text:\n")
print(output)
