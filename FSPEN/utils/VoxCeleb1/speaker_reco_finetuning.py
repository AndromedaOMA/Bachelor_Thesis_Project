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

# --- 1. Environment Stabilization ---
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 2. Setup Models & Device ---
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

predictor = EFSPENEnhancer("../../models/best_model_0.0613.pth", device=device)
if hasattr(predictor, 'model'):
    predictor.model.eval()

print("Loading Teacher model (Frozen)...")
teacher = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

print("Loading Student model (Trainable)...")
student = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
student.train()
for param in student.parameters():
    param.requires_grad = False
for param in student.mods.embedding_model.parameters():
    param.requires_grad = True

# --- 3. Training Prep ---
optimizer = optim.Adam(student.mods.embedding_model.parameters(), lr=5e-6)
criterion = torch.nn.CosineEmbeddingLoss()

# --- 4. Dataset Streaming ---
train_features = Features({"wav": Value("binary"), "__key__": Value("string"), "__url__": Value("string")})
val_features = Features({
    "audio": {"bytes": Value("binary"), "path": Value("string")},
    "id": Value("string"),
    "speaker_id": Value("string")
})

train_ds = load_dataset("Acouspike/Voxceleb1", split="train", streaming=True, features=train_features)
val_ds = load_dataset("asahi417/voxceleb1-test-split", split="test", streaming=True, features=val_features)
val_iterator = iter(val_ds)  # Keep an iterator open for validation chunks

# --- 5. Training & Early Stopping Config ---
num_epochs = 10
steps_per_epoch = 2000   # The model will see 32,000 total files (16 batch * 2000) per epoch
batch_size = 16
target_len = 64000
val_interval = 400       # Validate every 400 steps
val_steps = 20
patience = 5             # Stop if validation doesn't improve for 5 checks
best_val_loss = float('inf')
patience_counter = 0


def process_audio(audio_bytes, sr_orig=None):
    """Helper to process audio bytes to standard shape."""
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != 16000:
        wav = T.Resample(sr, 16000)(wav)
    wav = wav.to(device)
    if wav.shape[1] > target_len:
        wav = wav[:, :target_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
    return wav


accumulated_noisy = []
accumulated_enhanced = []
current_step = 0
print(f"Starting Training: {num_epochs} Epochs, {steps_per_epoch} Steps/Epoch...")

try:
    for epoch in range(num_epochs):
        print(f"\n========== EPOCH {epoch + 1}/{num_epochs} ==========")

        # We recreate the iterator at the start of each epoch to restart the stream
        train_ds = load_dataset("Acouspike/Voxceleb1", split="train", streaming=True, features=train_features)

        accumulated_noisy = []
        accumulated_enhanced = []
        step_in_epoch = 0

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}")

        for item in train_ds:
            if step_in_epoch >= steps_per_epoch:
                break  # End the epoch

            audio_bytes = item.get('wav')
            if not audio_bytes: continue

            try:
                wav = process_audio(audio_bytes)

                with torch.no_grad():
                    enhanced_wav = predictor.enhance_audio(wav)

                accumulated_noisy.append(wav.squeeze(0))
                accumulated_enhanced.append(enhanced_wav.squeeze((0, 1)).to(device))

                if len(accumulated_noisy) == batch_size:
                    # --- TRAINING PASS ---
                    optimizer.zero_grad()
                    batch_noisy = torch.stack(accumulated_noisy).to(device)
                    batch_enhanced = torch.stack(accumulated_enhanced).to(device)
                    rel_len = torch.ones(batch_size, device=device)

                    with torch.no_grad():
                        feats_t = teacher.mods.compute_features(batch_noisy)
                        feats_t = teacher.mods.mean_var_norm(feats_t, rel_len)
                        emb_t = teacher.mods.embedding_model(feats_t).squeeze(1)

                    feats_s = student.mods.compute_features(batch_enhanced)
                    feats_s = student.mods.mean_var_norm(feats_s, rel_len)
                    emb_s = student.mods.embedding_model(feats_s).squeeze(1)

                    target = torch.ones(batch_size).to(device)
                    loss = criterion(emb_s, emb_t, target)
                    loss.backward()
                    optimizer.step()

                    accumulated_noisy = []
                    accumulated_enhanced = []
                    step_in_epoch += 1
                    pbar.update(1)
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

                    # --- VALIDATION PHASE ---
                    if step_in_epoch % val_interval == 0:
                        student.eval()
                        val_loss_sum = 0.0
                        v_accum_noisy, v_accum_enh = [], []
                        v_steps_completed = 0

                        with torch.no_grad():
                            while v_steps_completed < val_steps:
                                try:
                                    v_item = next(val_iterator)
                                except StopIteration:
                                    val_iterator = iter(val_ds)
                                    v_item = next(val_iterator)

                                v_bytes = v_item.get('audio', {}).get('bytes')
                                if not v_bytes: continue

                                v_wav = process_audio(v_bytes)
                                v_enh = predictor.enhance_audio(v_wav)

                                v_accum_noisy.append(v_wav.squeeze(0))
                                v_accum_enh.append(v_enh.squeeze((0, 1)).to(device))

                                if len(v_accum_noisy) == batch_size:
                                    v_batch_noisy = torch.stack(v_accum_noisy).to(device)
                                    v_batch_enh = torch.stack(v_accum_enh).to(device)
                                    v_rel_len = torch.ones(batch_size, device=device)

                                    v_feats_t = teacher.mods.compute_features(v_batch_noisy)
                                    v_feats_t = teacher.mods.mean_var_norm(v_feats_t, v_rel_len)
                                    v_emb_t = teacher.mods.embedding_model(v_feats_t).squeeze(1)

                                    v_feats_s = student.mods.compute_features(v_batch_enh)
                                    v_feats_s = student.mods.mean_var_norm(v_feats_s, v_rel_len)
                                    v_emb_s = student.mods.embedding_model(v_feats_s).squeeze(1)

                                    v_target = torch.ones(batch_size).to(device)
                                    v_loss = criterion(v_emb_s, v_emb_t, v_target)
                                    val_loss_sum += v_loss.item()

                                    v_accum_noisy, v_accum_enh = [], []
                                    v_steps_completed += 1

                        avg_val_loss = val_loss_sum / val_steps

                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            torch.save(student.mods.embedding_model.state_dict(), "best_ecapa_adapted_efspen.pth")
                            pbar.write(f"--> Step {step_in_epoch}: New best model! Val Loss: {avg_val_loss:.4f}")
                        else:
                            patience_counter += 1
                            pbar.write(
                                f"--> Step {step_in_epoch}: No improvement. Val Loss: {avg_val_loss:.4f} (Patience: {patience_counter}/{patience})")

                        if patience_counter >= patience:
                            break  # Break validation interval loop

                        student.train()

            except Exception as e:
                continue

        pbar.close()

        # If patience exceeded during the inner loop, break the outer epoch loop too
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! Training halted at Epoch {epoch + 1}.")
            break

except Exception as e:
    print(f"\nTraining interrupted: {e}")

print(f"\nTraining complete. Best Validation Loss: {best_val_loss:.4f}")
