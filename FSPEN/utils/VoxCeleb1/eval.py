import io
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
from datasets import load_dataset, config, Features, Value
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.metric_stats import EER
from FSPEN.utils.VoxCeleb1.efspen_pred import EFSPENEnhancer

# 1. Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Global Configurations ---
config.HF_HUB_READ_TIMEOUT = 60
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load Enhancer Predictor
predictor = EFSPENEnhancer("../../models/best_model_0.0613.pth", device=device)

# Load Baseline Verification Model (for original noisy audio)
baseline_verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
baseline_verification.eval()
for param in baseline_verification.parameters():
    param.requires_grad = False

# Load Adapted Verification Model (for EFSPEN enhanced audio)
adapted_verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
# Load the newly fine-tuned weights
state_dict = torch.load("best_ecapa_adapted_efspen.pth", map_location=device, weights_only=True)
adapted_verification.mods.embedding_model.load_state_dict(state_dict)
adapted_verification.eval()
for param in adapted_verification.parameters():
    param.requires_grad = False

print("Baseline and Adapted models loaded successfully.")

# --- 2. Dataset Loading ---
custom_features = Features({
    "audio": {"bytes": Value("binary"), "path": Value("string")},
    "id": Value("string"),
    "speaker_id": Value("string")
})

ds = load_dataset(
    "asahi417/voxceleb1-test-split",
    split="test",
    streaming=True,
    features=custom_features
)

NUM_SPEAKERS = 20
SAMPLES_PER_SPK = 45
speaker_data = defaultdict(list)
found_speakers = []
found_speakers = []
resampler = None

print(f"Processing {NUM_SPEAKERS} speakers for Evaluation...")

# --- 3. Processing Loop ---
try:
    with torch.inference_mode():
        for item in ds:
            spk_id = item.get('speaker_id')
            audio_feature = item.get('audio')

            if spk_id is None or audio_feature is None:
                continue

            if spk_id not in found_speakers:
                if len(found_speakers) < NUM_SPEAKERS:
                    found_speakers.append(spk_id)
                else:
                    continue

            if spk_id in found_speakers and len(speaker_data[spk_id]) < SAMPLES_PER_SPK:
                audio_bytes = audio_feature.get('bytes')
                if audio_bytes is None:
                    continue

                wav, original_sr = torchaudio.load(io.BytesIO(audio_bytes))

                if original_sr != 16000:
                    if resampler is None or resampler.orig_freq != original_sr:
                        resampler = T.Resample(orig_freq=original_sr, new_freq=16000)
                    wav = resampler(wav)

                wav = wav.to(device)

                target_len = 64000
                if wav.shape[1] > target_len:
                    wav = wav[:, :target_len]
                elif wav.shape[1] < target_len:
                    wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))

                # --- 1. Baseline Pathway ---
                emb_noisy = baseline_verification.encode_batch(wav).squeeze().cpu()

                # --- 2. Adapted Pathway ---
                enhanced_wav = predictor.enhance_audio(wav)
                emb_enh_adapted = adapted_verification.encode_batch(enhanced_wav.to(device)).squeeze().cpu()

                # Store Normalized Embeddings
                speaker_data[spk_id].append({
                    "Baseline Noisy": torch.nn.functional.normalize(emb_noisy, dim=-1),
                    "Adapted Enhanced": torch.nn.functional.normalize(emb_enh_adapted, dim=-1)
                })
                print(f"Captured: Spk {spk_id} | Sample {len(speaker_data[spk_id])}/{SAMPLES_PER_SPK}")

except Exception as e:
    print(f"\nProcessing error: {e}")

# --- 4. EER Calculation & Final Results ---
if len(speaker_data) < 2:
    print(f"Insufficient data. Collected {len(speaker_data)} speakers.")
else:
    print("\nCalculating EER Results...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle('VoxCeleb1 Performance Comparison', fontsize=16, fontweight='bold')

    results_summary = {}
    modes = ["Baseline Noisy", "Adapted Enhanced"]

    for idx, mode in enumerate(modes):
        pos_scores, neg_scores = [], []
        spks = list(speaker_data.keys())

        # Target Scores
        for spk in spks:
            embs = [d[mode] for d in speaker_data[spk]]
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    pos_scores.append(torch.dot(embs[i], embs[j]).item())

        # Impostor Scores
        for i in range(len(spks)):
            for j in range(i + 1, len(spks)):
                e1 = speaker_data[spks[i]][0][mode]
                e2 = speaker_data[spks[j]][0][mode]
                neg_scores.append(torch.dot(e1, e2).item())

        if pos_scores and neg_scores:
            eer, _ = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))
            results_summary[mode] = eer * 100

            ax = axes[idx]
            ax.hist(neg_scores, bins=30, alpha=0.5, label='Impostors', color='red', density=True)
            ax.hist(pos_scores, bins=30, alpha=0.5, label='Targets', color='blue', density=True)
            ax.set_title(f"{mode}\nEER: {eer * 100:.2f}%")
            ax.set_xlabel("Cosine Similarity Score")
            ax.legend()

    print("\n" + "=" * 35)
    for k, v in results_summary.items():
        print(f"{k} EER: {v:.2f}%")
    print("=" * 35)

    plt.tight_layout()
    plt.show()
