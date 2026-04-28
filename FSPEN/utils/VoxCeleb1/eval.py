# # import torch
# # import torchaudio
# # import torchaudio.transforms as T
# # import matplotlib.pyplot as plt
# # from collections import defaultdict
# # from datasets import load_dataset, config
# # from speechbrain.inference.speaker import SpeakerRecognition
# # from speechbrain.utils.metric_stats import EER
# # from FSPEN.utils.VoxCeleb1.efspen_pred import EFSPENEnhancer
# #
# # # --- 1. Configurations ---
# # config.HF_HUB_READ_TIMEOUT = 30  # Allow slow stream initialization
# # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# #
# # predictor = EFSPENEnhancer("../../models/best_model_0.0613.pth", device=device)
# # verification = SpeakerRecognition.from_hparams(
# #     source="speechbrain/spkrec-ecapa-voxceleb",
# #     run_opts={"device": device}
# # )
# # verification.eval()
# #
# # # --- 2. Dataset Loading (Manual Resampling Bypass) ---
# # print("Loading VoxCeleb1 streaming dataset...")
# # ds = load_dataset("ProgramComputer/voxceleb", name="default", split="test", streaming=True)
# #
# # NUM_SPEAKERS = 5
# # SAMPLES_PER_SPK = 4
# # speaker_data = defaultdict(list)
# # found_speakers = set()
# # resampler = None
# #
# # print(f"Processing {NUM_SPEAKERS} speakers...")
# #
# # # --- 3. Robust Processing Loop ---
# # try:
# #     for item in ds:
# #         # Check for speaker identity and audio stream
# #         spk_id = item.get('label') or item.get('speaker_id')
# #         audio_data = item.get('audio')
# #
# #         if not spk_id or not audio_data:
# #             continue
# #
# #         if spk_id not in found_speakers:
# #             if len(found_speakers) >= NUM_SPEAKERS: break
# #             found_speakers.add(spk_id)
# #
# #         if len(speaker_data[spk_id]) < SAMPLES_PER_SPK:
# #             # Manual Resample Workaround for TypeError
# #             audio_array = audio_data.get('array')
# #             original_sr = audio_data.get('sampling_rate')
# #
# #             if audio_array is None: continue
# #             wav = torch.from_numpy(audio_array).float().unsqueeze(0)
# #
# #             if original_sr != 16000:
# #                 if resampler is None or resampler.orig_freq != original_sr:
# #                     resampler = T.Resample(orig_freq=original_sr, new_freq=16000)
# #                 wav = resampler(wav)
# #
# #             wav = wav.to(device)
# #
# #             # Standardize length to 4s (64,000 samples)
# #             target_len = 64000
# #             if wav.shape[1] > target_len:
# #                 wav = wav[:, :target_len]
# #             elif wav.shape[1] < target_len:
# #                 wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
# #
# #             with torch.no_grad():
# #                 emb_noisy = verification.encode_batch(wav).squeeze().cpu()
# #                 enhanced_wav = predictor.enhance_audio(wav)
# #                 emb_enh = verification.encode_batch(enhanced_wav.to(device)).squeeze().cpu()
# #
# #             speaker_data[spk_id].append({
# #                 "noisy": torch.nn.functional.normalize(emb_noisy, dim=-1),
# #                 "enhanced": torch.nn.functional.normalize(emb_enh, dim=-1)
# #             })
# #             print(f"Processed sample for {spk_id} ({len(speaker_data[spk_id])}/{SAMPLES_PER_SPK})")
# #
# # except Exception as e:
# #     print(f"\nStreaming Error/Interruption: {e}")
# #
# # # --- 4. EER Calculation with Empty Data Guard ---
# # if len(speaker_data) < 2:
# #     print("Insufficient data collected. Check connection or increase NUM_SPEAKERS.")
# # else:
# #     print("\nGenerating Results...")
# #     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# #     for idx, mode in enumerate(["noisy", "enhanced"]):
# #         pos_scores, neg_scores = [], []
# #         spks = list(speaker_data.keys())
# #
# #         for spk in spks:
# #             embs = [d[mode] for d in speaker_data[spk]]
# #             for i in range(len(embs)):
# #                 for j in range(i + 1, len(embs)):
# #                     pos_scores.append(torch.dot(embs[i], embs[j]).item())
# #
# #         for i in range(len(spks)):
# #             for j in range(i + 1, len(spks)):
# #                 e1 = speaker_data[spks[i]][0][mode]
# #                 e2 = speaker_data[spks[j]][0][mode]
# #                 neg_scores.append(torch.dot(e1, e2).item())
# #
# #         if pos_scores and neg_scores:
# #             eer, _ = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))
# #             ax = axes[idx]
# #             ax.hist(neg_scores, bins=30, alpha=0.5, label='Impostors', color='red', density=True)
# #             ax.hist(pos_scores, bins=30, alpha=0.5, label='Targets', color='blue', density=True)
# #             ax.set_title(f"{mode.capitalize()} (VoxCeleb1)\nEER: {eer * 100:.2f}%")
# #             ax.legend()
# #
# #     plt.tight_layout()
# #     plt.show()
#
# import io
# import torch
# import torchaudio
# import torchaudio.transforms as T
# import matplotlib.pyplot as plt
# import warnings
# from collections import defaultdict
# from datasets import load_dataset, config, Features, Value
# from speechbrain.inference.speaker import SpeakerRecognition
# from speechbrain.utils.metric_stats import EER
# from FSPEN.utils.VoxCeleb1.efspen_pred import EFSPENEnhancer
#
# # 1. Silence FutureWarning from SpeechBrain for cleaner output
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# # --- 1. Global Configurations ---
# config.HF_HUB_READ_TIMEOUT = 60
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Load Predictor and Verification Models
# predictor = EFSPENEnhancer("../../models/best_model_0.0613.pth", device=device)
# verification = SpeakerRecognition.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb",
#     run_opts={"device": device}
# )
#
#
# verification.eval()
# for param in verification.parameters():
#     param.requires_grad = False
#
# # --- 2. Dataset Loading (Torchcodec Bypass) ---
# print("Loading asahi417/voxceleb1-test-split dataset...")
#
# # We define the schema manually. By setting 'audio' to a dict of binary 'bytes',
# # we prevent the 'datasets' library from trying to use torchcodec for decoding.
# custom_features = Features({
#     "audio": {"bytes": Value("binary"), "path": Value("string")},
#     "id": Value("string"),
#     "speaker_id": Value("string")
# })
#
# ds = load_dataset(
#     "asahi417/voxceleb1-test-split",
#     split="test",
#     streaming=True,
#     features=custom_features  # Forces the bypass
# )
#
# NUM_SPEAKERS = 30
# SAMPLES_PER_SPK = 50
# speaker_data = defaultdict(list)
# found_speakers = []
# resampler = None
#
# print(f"Processing {NUM_SPEAKERS} speakers using manual decoding...")
#
# # --- 3. Processing Loop ---
# try:
#     for item in ds:
#         # Use the keys from your screenshot: 'audio' and 'speaker_id'
#         spk_id = item.get('speaker_id')
#         audio_feature = item.get('audio')
#
#         if spk_id is None or audio_feature is None:
#             continue
#
#         if spk_id not in found_speakers:
#             if len(found_speakers) < NUM_SPEAKERS:
#                 found_speakers.append(spk_id)
#             else:
#                 continue
#
#         if spk_id in found_speakers and len(speaker_data[spk_id]) < SAMPLES_PER_SPK:
#             # Extract raw binary bytes
#             audio_bytes = audio_feature.get('bytes')
#             if audio_bytes is None:
#                 continue
#
#             # Decode manually using torchaudio (No torchcodec needed)
#             wav, original_sr = torchaudio.load(io.BytesIO(audio_bytes))
#
#             # Resample to 16kHz
#             if original_sr != 16000:
#                 if resampler is None or resampler.orig_freq != original_sr:
#                     resampler = T.Resample(orig_freq=original_sr, new_freq=16000)
#                 wav = resampler(wav)
#
#             wav = wav.to(device)
#
#             # Standardize length to 4s (16000 * 4 = 64000 samples)
#             target_len = 64000
#             if wav.shape[1] > target_len:
#                 wav = wav[:, :target_len]
#             elif wav.shape[1] < target_len:
#                 wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
#
#             with torch.no_grad():
#                 # Encode Noisy Embedding
#                 emb_noisy = verification.encode_batch(wav).squeeze().cpu()
#
#                 # EFSPEN Enhancement
#                 enhanced_wav = predictor.enhance_audio(wav)
#
#                 # Encode Enhanced Embedding
#                 emb_enh = verification.encode_batch(enhanced_wav.to(device)).squeeze().cpu()
#
#             # Store Normalized Embeddings for EER
#             speaker_data[spk_id].append({
#                 "noisy": torch.nn.functional.normalize(emb_noisy, dim=-1),
#                 "enhanced": torch.nn.functional.normalize(emb_enh, dim=-1)
#             })
#             print(f"Captured: Spk {spk_id} | Sample {len(speaker_data[spk_id])}/{SAMPLES_PER_SPK}")
#
# except Exception as e:
#     print(f"\nProcessing error: {e}")
#
# # torch.cuda.empty_cache()
# # print("Starting EER calculation...")
#
# # --- 4. EER Calculation & Final Results ---
# if len(speaker_data) < 2:
#     print(f"Insufficient data. Collected {len(speaker_data)} speakers. Check internet or increase NUM_SPEAKERS.")
# else:
#     print("\nCalculating EER Results...")
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#     plt.suptitle('VoxCeleb1 dataset', fontsize=16, fontweight='bold')
#     results_summary = {}
#
#     for idx, mode in enumerate(["noisy", "enhanced"]):
#         pos_scores, neg_scores = [], []
#         spks = list(speaker_data.keys())
#
#         # Target Scores (Same Speaker)
#         for spk in spks:
#             embs = [d[mode] for d in speaker_data[spk]]
#             for i in range(len(embs)):
#                 for j in range(i + 1, len(embs)):
#                     pos_scores.append(torch.dot(embs[i], embs[j]).item())
#
#         # Impostor Scores (Different Speakers)
#         for i in range(len(spks)):
#             for j in range(i + 1, len(spks)):
#                 e1 = speaker_data[spks[i]][0][mode]
#                 e2 = speaker_data[spks[j]][0][mode]
#                 neg_scores.append(torch.dot(e1, e2).item())
#
#         if pos_scores and neg_scores:
#             eer, _ = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))
#             results_summary[mode] = eer * 100
#
#             ax = axes[idx]
#             ax.hist(neg_scores, bins=30, alpha=0.5, label='Impostors', color='red', density=True)
#             ax.hist(pos_scores, bins=30, alpha=0.5, label='Targets', color='blue', density=True)
#             ax.set_title(f"{mode.capitalize()}\nEER: {eer * 100:.2f}%")
#             ax.set_xlabel("Cosine Similarity Score")
#             ax.legend()
#
#     print("\n" + "=" * 30)
#     for k, v in results_summary.items():
#         print(f"{k.capitalize()} EER: {v:.2f}%")
#     print("=" * 30)
#
#     plt.tight_layout()
#     plt.show()


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

# 1. Suppress warnings for a cleaner console
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Global Configurations ---
config.HF_HUB_READ_TIMEOUT = 60
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load Predictor
predictor = EFSPENEnhancer("../../models/best_model_0.0613.pth", device=device)

# Load Verification Model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
state_dict = torch.load("ecapa_adapted_efspen.pth", map_location=device)
verification.mods.embedding_model.load_state_dict(state_dict)

# C. Set to evaluation mode
print("Adapted model loaded successfully.")

# --- SET VERIFICATION TO INFERENCE ONLY ---
verification.eval()  # Sets Dropout and BatchNorm to eval mode
for param in verification.parameters():
    param.requires_grad = False  # Freezes weights to save memory/prevent updates
# ------------------------------------------

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

NUM_SPEAKERS = 30
SAMPLES_PER_SPK = 50
speaker_data = defaultdict(list)
found_speakers = []
resampler = None

print(f"Processing {NUM_SPEAKERS} speakers in strict Inference Mode...")

# --- 3. Processing Loop ---
try:
    # Use torch.inference_mode() for the entire loop for maximum speed/memory efficiency
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

                # Standardize length to 4s
                target_len = 64000
                if wav.shape[1] > target_len:
                    wav = wav[:, :target_len]
                elif wav.shape[1] < target_len:
                    wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))

                # --- Inference Only Execution ---
                # Encode Noisy Embedding
                emb_noisy = verification.encode_batch(wav).squeeze().cpu()

                # EFSPEN Enhancement
                enhanced_wav = predictor.enhance_audio(wav)

                # Encode Enhanced Embedding
                emb_enh = verification.encode_batch(enhanced_wav.to(device)).squeeze().cpu()

                # Store Normalized Embeddings
                speaker_data[spk_id].append({
                    "noisy": torch.nn.functional.normalize(emb_noisy, dim=-1),
                    "enhanced": torch.nn.functional.normalize(emb_enh, dim=-1)
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
    plt.suptitle('VoxCeleb1 Performance (Inference Mode)', fontsize=16, fontweight='bold')

    results_summary = {}

    for idx, mode in enumerate(["noisy", "enhanced"]):
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
            ax.set_title(f"{mode.capitalize()}\nEER: {eer * 100:.2f}%")
            ax.set_xlabel("Cosine Similarity Score")
            ax.legend()

    print("\n" + "=" * 30)
    for k, v in results_summary.items():
        print(f"{k.capitalize()} EER: {v:.2f}%")
    print("=" * 30)

    plt.tight_layout()
    plt.show()
