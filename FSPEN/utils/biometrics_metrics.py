"""VoiceBankDEMAND"""

import os
import random
import sys
import io
from collections import defaultdict
from unittest.mock import MagicMock

# 1. BYPASS SPEECHBRAIN 1.0 INTEGRATION BUGS
for module in ["k2", "flair"]:
    sys.modules[module] = MagicMock()

import torch
import torchaudio
import matplotlib.pyplot as plt
# Import 'Value' and 'Features' to strip Audio metadata
from datasets import load_dataset, Audio, Value, Features
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.metric_stats import EER

# Silence warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Define folders
enhanced_dir = "../evaluation/outputs/4sec_samples_mha/enhanced"

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
verification.eval()
for param in verification.parameters():
    param.requires_grad = False

# Initialize the Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.suptitle('VoiceBank-DEMAND dataset', fontsize=16, fontweight='bold')
results_summary = {}

# --- A. PROCESS NOISY (TRAINING DATASET - FIRST 2 SPEAKERS) ---
print("\nProcessing Noisy Training samples (First 2 Speakers)...")

# 1. Load the dataset
ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split="train")

# 2. THE ULTIMATE FIX: Strip the 'Audio' type metadata completely
# This redefines the columns as simple nested dictionaries, so torchcodec is never called.
new_features = ds.features.copy()
new_features["noisy"] = {"bytes": Value("binary"), "path": Value("string")}
new_features["clean"] = {"bytes": Value("binary"), "path": Value("string")}
ds = ds.cast(new_features)

speaker_map_noisy = defaultdict(list)
embeddings_noisy = {}
found_speakers = []

for i in range(len(ds)):
    item = ds[i]
    speaker_id = item['id'].split('_')[0]

    if speaker_id not in found_speakers:
        if len(found_speakers) < 2:
            found_speakers.append(speaker_id)
            print(f"Found speaker: {speaker_id}")
        else:
            continue

    if speaker_id in found_speakers:
        # Load waveform from raw binary bytes
        raw_bytes = item['noisy']['bytes']
        signal, fs = torchaudio.load(io.BytesIO(raw_bytes))

        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)

        with torch.no_grad():
            emb = verification.encode_batch(signal.to(device))
            embeddings_noisy[item['id']] = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu()
            speaker_map_noisy[speaker_id].append(item['id'])

# --- B. PROCESS ENHANCED (LOCAL FOLDER) ---
print(f"\nProcessing Enhanced samples from {enhanced_dir}...")
speaker_map_enh = defaultdict(list)
embeddings_enh = {}

if os.path.exists(enhanced_dir):
    all_enh_files = [f for f in os.listdir(enhanced_dir) if f.endswith('.wav')]
    for filename in all_enh_files:
        file_path = os.path.join(enhanced_dir, filename)
        signal, fs = torchaudio.load(file_path)
        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)

        with torch.no_grad():
            emb = verification.encode_batch(signal.to(device))
            embeddings_enh[filename] = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu()

            # Grouping logic: enhanced_{batch}_{idx}.wav -> use index as speaker ID
            parts = filename.split('_')
            spk_idx = parts[1]
            speaker_map_enh[spk_idx].append(filename)

# --- 5. GENERATE SCORES AND PLOT ---
eval_tasks = [
    ("Noisy", speaker_map_noisy, embeddings_noisy, 0),
    ("Enhanced", speaker_map_enh, embeddings_enh, 1)
]

for label, spk_map, emb_dict, plt_idx in eval_tasks:
    if len(spk_map) < 2:
        print(f"Skipping {label}: Need 2 speakers.")
        continue

    pos_scores, neg_scores = [], []
    for spk, files in spk_map.items():
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pos_scores.append(torch.dot(emb_dict[files[i]], emb_dict[files[j]]).item())

    spks = list(spk_map.keys())
    for i in range(len(spks)):
        for j in range(i + 1, len(spks)):
            f1 = random.choice(spk_map[spks[i]])
            f2 = random.choice(spk_map[spks[j]])
            neg_scores.append(torch.dot(emb_dict[f1], emb_dict[f2]).item())

    eer, _ = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))
    results_summary[label] = eer * 100

    ax = axes[plt_idx]
    ax.hist(neg_scores, bins=50, alpha=0.5, label='Impostors', color='red', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.5, label='Targets', color='blue', density=True)
    ax.set_title(f"{label}\nEER: {eer * 100:.2f}%")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper right')

print("\n" + "=" * 30)
for k, v in results_summary.items():
    print(f"{k} EER: {v:.2f}%")
print("=" * 30)

plt.tight_layout()
plt.show()

"""LibriSpeech"""

import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.metric_stats import EER

# Paths
base_dir = "../evaluation/outputs/librispeech_mha"
# base_dir = "../evaluation/outputs/4sec_samples_mha"
noisy_dir = os.path.join(base_dir, "noisy")
enhanced_dir = os.path.join(base_dir, "enhanced")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
verification.eval()
for param in verification.parameters():
    param.requires_grad = False

def voicebank_get_embeddings(folder_path):
    speaker_map = defaultdict(list)
    emb_dict = {}
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    print(f"Processing {len(files)} files from {folder_path}...")
    for filename in files:
        path = os.path.join(folder_path, filename)
        signal, fs = torchaudio.load(path)
        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)

        with torch.no_grad():
            emb = verification.encode_batch(signal.to(device))
            # Normalize for cosine similarity
            norm_emb = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu()
            emb_dict[filename] = norm_emb

            # Extract speaker ID (e.g., noisy_p232_001.wav -> p232)
            parts = filename.split('_')
            spk_id = parts[1]
            speaker_map[spk_id].append(filename)

    return speaker_map, emb_dict


def librespeech_get_embeddings(folder_path):
    speaker_map = defaultdict(list)
    emb_dict = {}
    files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.flac'))]

    print(f"Processing {len(files)} files from {folder_path}...")
    for filename in files:
        path = os.path.join(folder_path, filename)
        signal, fs = torchaudio.load(path)
        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)

        with torch.no_grad():
            emb = verification.encode_batch(signal.to(device))
            norm_emb = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu()
            emb_dict[filename] = norm_emb

            # LIBRISPEECH FIX: Split by '-' instead of '_'
            # Example: '1272-128104-0000.wav' -> parts[0] is '1272'
            parts = filename.split('-')
            if len(parts) > 1:
                spk_id = parts[0]  # For LibriSpeech, the first part is the Speaker ID
                speaker_map[spk_id].append(filename)
            else:
                # Fallback for other naming conventions like 'enhanced_p232_001.wav'
                parts_alt = filename.split('_')
                spk_id = parts_alt[1] if len(parts_alt) > 1 else "unknown"
                speaker_map[spk_id].append(filename)

    return speaker_map, emb_dict


# 1. Generate Embeddings
noisy_map, noisy_embs = librespeech_get_embeddings(noisy_dir)
enh_map, enh_embs = librespeech_get_embeddings(enhanced_dir)

# 2. Evaluation Loop
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.suptitle('LibriSpeech dataset', fontsize=16, fontweight='bold')
eval_tasks = [
    ("Noisy Samples", noisy_map, noisy_embs, 0),
    ("Enhanced Samples", enh_map, enh_embs, 1)
]

for label, spk_map, emb_dict, plt_idx in eval_tasks:
    pos_scores, neg_scores = [], []

    # Positive pairs (Same speaker)
    for spk, files in spk_map.items():
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pos_scores.append(torch.dot(emb_dict[files[i]], emb_dict[files[j]]).item())

    # Negative pairs (Different speakers)
    spks = list(spk_map.keys())
    for i in range(len(spks)):
        for j in range(i + 1, len(spks)):
            # Randomly pair samples from different speakers
            f1 = random.choice(spk_map[spks[i]])
            f2 = random.choice(spk_map[spks[j]])
            neg_scores.append(torch.dot(emb_dict[f1], emb_dict[f2]).item())

    eer, _ = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))

    ax = axes[plt_idx]
    ax.hist(neg_scores, bins=50, alpha=0.5, label='Impostors', color='red', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.5, label='Targets', color='blue', density=True)
    ax.set_title(f"{label}\nEER: {eer * 100:.2f}%")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
