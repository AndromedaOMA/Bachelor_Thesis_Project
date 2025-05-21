from pesq import pesq
from pystoi import stoi
import os
import soundfile as sf
from torch.utils.data import DataLoader

from FSPEN.data.voicebank_demand_16K import VoiceBankDEMAND, configs

pesq_scores = []
stoi_scores = []

test_dataset = VoiceBankDEMAND(device, configs, mode="test")
test_loader = DataLoader(test_dataset, batch_size=configs.batch_size)

for i in range(len(test_dataset)):
    clean, _ = sf.read(f'outputs/clean_{i}.wav')
    enhanced, _ = sf.read(f'outputs/enhanced_{i}.wav')

    # Ensure same length
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    pesq_score = pesq(16000, clean, enhanced, 'wb')  # 'wb' = wideband
    stoi_score = stoi(clean, enhanced, 16000, extended=False)

    pesq_scores.append(pesq_score)
    stoi_scores.append(stoi_score)

print(f"Average PESQ: {sum(pesq_scores) / len(pesq_scores):.4f}")
print(f"Average STOI: {sum(stoi_scores) / len(stoi_scores):.4f}")
