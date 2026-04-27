import os
import io
import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset, Audio


def extract_first_two_speakers(output_base_dir="extracted_speakers"):
    # 1. Load the training dataset without decoding to check metadata quickly
    print("Loading dataset...")
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split="train")
    # Ensure we get raw bytes to avoid torchcodec issues
    ds = ds.cast_column("noisy", Audio(decode=False))

    unique_speakers = []
    speaker_samples = {}  # Store indices of samples for the chosen speakers

    print("Searching for the first 2 unique speakers...")
    for idx, example in enumerate(ds):
        # Path format is usually 'p226_001.wav'
        file_path = example['noisy']['path']
        speaker_id = file_path.split('_')[0]

        if speaker_id not in unique_speakers:
            unique_speakers.append(speaker_id)
            speaker_samples[speaker_id] = []
            print(f"Found speaker: {speaker_id}")

        # Collect samples for the speakers we care about
        if speaker_id in unique_speakers:
            speaker_samples[speaker_id].append(idx)

        # Stop searching once we have 2 speakers
        if len(unique_speakers) == 2:
            # We continue the loop if we want ALL samples of these 2 speakers,
            # but if we just want to find WHO they are, we can break or limit search.
            # Let's stop searching after a reasonable range to save time.
            if idx > 1000:  # Adjust if speakers are spread far apart
                break

    # 2. Create directories and save audio
    for spk_id in unique_speakers:
        spk_dir = os.path.join(output_base_dir, spk_id)
        os.makedirs(spk_dir, exist_ok=True)

        print(f"Saving samples for {spk_id}...")
        # Save only the first 5 samples per speaker to keep it quick
        for sample_idx in speaker_samples[spk_id][:5]:
            example = ds[sample_idx]
            raw_bytes = example['noisy']['bytes']
            file_name = example['noisy']['path']

            # Use torchaudio + io.BytesIO to decode (bypassing torchcodec)
            waveform, sr = torchaudio.load(io.BytesIO(raw_bytes))

            # Save to disk
            output_path = os.path.join(spk_dir, file_name)
            sf.write(output_path, waveform.squeeze().numpy(), sr)

    print(f"Extraction complete! Files saved in: {output_base_dir}")


if __name__ == "__main__":
    extract_first_two_speakers()
