import sys
import io
import torch
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import warnings
from tqdm import tqdm
from datasets import load_dataset, Features, Value
from unittest.mock import MagicMock

# --- CRITICAL ENVIRONMENT STABILIZATION BLOCK ---
# Mocking modules to prevent k2/dynamo crashes in certain environments
mock_modules = [
    "k2", "flair", "flair.data", "numba", "speechbrain.wordemb",
    "speechbrain.integrations", "speechbrain.integrations.nlp",
    "speechbrain.integrations.k2_fsa", "speechbrain.integrations.numba",
    "speechbrain.integrations.numba.transducer_loss",
    "speechbrain.integrations.huggingface",
    "speechbrain.integrations.huggingface.wordemb"
]
for module in mock_modules:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

from speechbrain.inference.speaker import SpeakerRecognition
from FSPEN.utils.VoxCeleb1.efspen_pred import EFSPENEnhancer

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Setup Models & Device ---
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# A. Enhancer (Inference only)
predictor = EFSPENEnhancer("../../models/best_model_0.0613.pth", device=device)
if hasattr(predictor, 'model'):
    predictor.model.eval()

# B. Verification Model (For Fine-Tuning)
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
verification.train()

# --- 2. Freeze Early Layers ---
for name, param in verification.mods.embedding_model.named_parameters():
    if "blocks.0" in name or "blocks.1" in name:
        print("Freezing early blocks of the embedding model...")
        param.requires_grad = False
    else:
        param.requires_grad = True

# --- 3. Training Prep ---
optimizer = optim.Adam(filter(lambda p: p.requires_grad, verification.parameters()), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# --- 4. Dataset Streaming ---
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

# --- 5. Speaker ID Mapping ---
speaker_map = {}


def get_label(spk_id):
    if spk_id not in speaker_map:
        speaker_map[spk_id] = len(speaker_map)
    return speaker_map[spk_id]


# --- 6. Fine-Tuning Loop ---
num_steps = 500
current_step = 0
batch_size = 8  # Accumulate samples for a real batch
accumulated_wavs = []
accumulated_labels = []

print(f"Starting Domain Adaptation with batch size {batch_size}...")
pbar = tqdm(total=num_steps)

for item in ds:
    if current_step >= num_steps: break

    spk_id = item.get('speaker_id')
    audio_bytes = item['audio'].get('bytes')

    if audio_bytes and spk_id:
        try:
            # 1. Load and Enhance (Single Sample)
            wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
            if sr != 16000: wav = T.Resample(sr, 16000)(wav)
            wav = wav.to(device)

            # Pad/Clip to consistent length
            if wav.shape[1] > 32000:
                wav = wav[:, :32000]
            else:
                wav = torch.nn.functional.pad(wav, (0, 32000 - wav.shape[1]))

            with torch.no_grad():
                enhanced_wav = predictor.enhance_audio(wav)
                enhanced_wav = enhanced_wav.squeeze(1).to(device)  # Shape: [1, 32000]

            accumulated_wavs.append(enhanced_wav)
            accumulated_labels.append(get_label(spk_id))

            # 2. Process as Batch
            if len(accumulated_wavs) == batch_size:
                optimizer.zero_grad()

                # Stack to [Batch, Time]
                batch_wav = torch.cat(accumulated_wavs, dim=0)
                batch_labels = torch.LongTensor(accumulated_labels).to(device)

                # Feature Extraction & Prediction
                feats = verification.mods.compute_features(batch_wav)
                rel_len = torch.ones(batch_size, device=device)
                feats = verification.mods.mean_var_norm(feats, rel_len)

                embeddings = verification.mods.embedding_model(feats)
                predictions = verification.mods.classifier(embeddings)

                # Loss & Backprop
                loss = criterion(predictions.squeeze(1), batch_labels)
                loss.backward()
                optimizer.step()

                # Cleanup
                accumulated_wavs = []
                accumulated_labels = []
                current_step += 1
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.4f}")

        except Exception as e:
            print(f"\nError at step {current_step}: {e}")
            continue

pbar.close()

# --- 7. Save ---
print("Saving adapted model...")
torch.save(verification.mods.embedding_model.state_dict(), "ecapa_adapted_efspen.pth")