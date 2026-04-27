import os
import torch
import torchaudio
import soundfile as sf
import io
from tqdm import tqdm
from datasets import load_dataset, Audio  # Added Audio for casting
from FSPEN.configs.train_configs import TrainConfig
from FSPEN.models.efspen import FullSubPathExtension


def prepare_initial_hidden_state(batch, num_bands, num_modules, groups, inter_hidden_size, device):
    return [
        [torch.zeros(1, batch * num_bands, inter_hidden_size // groups).to(device=device)
         for _ in range(groups)]
        for _ in range(num_modules)
    ]


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = TrainConfig()

    # 1. Load Model with security fix (weights_only=True)
    model = FullSubPathExtension(configs).to(device)
    state_path = "../models/best_model_0.0613.pth"
    model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Setup Directories
    out_dir = 'outputs/librispeech_mha'
    for folder in ['enhanced', 'clean', 'noisy']:
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)

    # 3. Load Dataset & Disable Automatic Decoding
    ds = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    # This tells the dataset to give us raw bytes instead of trying to use torchcodec
    ds = ds.cast_column("audio", Audio(decode=False))

    print("Processing started. Decoding audio manually via torchaudio...")

    limit = 100

    with torch.no_grad():
        for i, item in enumerate(tqdm(ds, desc="Enhancing Samples")):
            if i >= limit:
                break

            # --- A. MANUAL DECODING (The Fix) ---
            # Extract bytes from the 'audio' dictionary and load with torchaudio
            audio_bytes = item["audio"]["bytes"]
            clean_wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
            clean_wav = clean_wav.to(device)

            # Ensure 16kHz if LibriSpeech varies (usually 16k, but safe to check)
            if sr != 16000:
                clean_wav = torchaudio.transforms.Resample(sr, 16000).to(device)(clean_wav)

            # Add synthetic noise
            noisy_wav = clean_wav + 0.01 * torch.randn_like(clean_wav)

            # --- B. SPECTRUM PREPARATION ---
            complex_spec = torch.stft(
                noisy_wav, n_fft=512, hop_length=256,
                window=torch.hamming_window(512).to(device), return_complex=True
            )

            amplitude = torch.abs(complex_spec).permute(0, 2, 1).unsqueeze(2)
            amplitude = (amplitude - amplitude.mean()) / (amplitude.std() + 1e-5)
            complex_in = torch.view_as_real(complex_spec).permute(0, 2, 3, 1)

            # --- C. INFERENCE ---
            hidden_state = prepare_initial_hidden_state(1, 32, 3, 8, 16, device)
            enhanced_spec_real, _ = model(complex_in, amplitude, hidden_state)

            # --- D. RECONSTRUCTION ---
            recon_spec = torch.complex(enhanced_spec_real[:, :, 0, :], enhanced_spec_real[:, :, 1, :]).permute(0, 2, 1)
            enhanced_wav = torch.istft(
                recon_spec, n_fft=512, hop_length=256, win_length=512,
                window=torch.hann_window(512).to(device)
            )

            # --- E. SAVE ---
            file_id = f"{item['speaker_id']}-{item['chapter_id']}-{item['id']}"

            enhanced_np = enhanced_wav.cpu().numpy().squeeze()
            clean_np = clean_wav.cpu().numpy().squeeze()
            noisy_np = noisy_wav.cpu().numpy().squeeze()

            sf.write(f'{out_dir}/enhanced/{file_id}.wav', enhanced_np, 16000)
            sf.write(f'{out_dir}/clean/{file_id}.wav', clean_np, 16000)
            sf.write(f'{out_dir}/noisy/{file_id}.wav', noisy_np, 16000)

    print(f"\nDone! Files saved to {out_dir}")