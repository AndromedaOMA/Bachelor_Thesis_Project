import json
import torch
import torch.optim as optim
from thop import profile, clever_format
from configs.train_configs import TrainConfig
from models.fspen import FullSubPathExtension
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import time


class AudioDataset(Dataset):
    def __init__(self, configs):
        # Încărcarea seturilor de date
        self.data_files = {
            'train': [
                '../../VoiceBank_DEMAND_16k/train/0000.parquet',
                '../../VoiceBank_DEMAND_16k/train/0001.parquet',
                '../../VoiceBank_DEMAND_16k/train/0002.parquet',
                '../../VoiceBank_DEMAND_16k/train/0003.parquet',
                '../../VoiceBank_DEMAND_16k/train/0004.parquet',
            ],
            'test': [
                '../../VoiceBank_DEMAND_16k/test/0000.parquet'
            ]}
        self.configs = configs
        self.dataset = load_dataset("parquet", data_files=self.data_files, streaming=True)
        self.train_batches = self.batches_generator(self.dataset['train'])
        self.test_batches = self.batches_generator(self.dataset['test'])

    def __len__(self):
        return len(self.dataset)

    def batches_generator(self, data):
        """YIELD batch-urile procesate cu STFT"""
        buffer = []

        for example in data.shuffle(buffer_size=5000, seed=34):
            buffer.append(example)

            # Delimitarea batch-urilor
            if len(buffer) == self.configs.batch_size:
                batch = buffer[:self.configs.batch_size]
                buffer = buffer[self.configs.batch_size:]

                # Convertim întregul batch în STFT
                stft_batch_clean = []
                stft_batch_noisy = []
                for sample in batch:
                    clean_wav = torch.tensor(sample['clean']['array'], dtype=torch.float32).unsqueeze(0)
                    noisy_wav = torch.tensor(sample['noisy']['array'], dtype=torch.float32).unsqueeze(0)

                    # Convertim semnalul curat și zgomotos în STFT
                    clean_complex_spectrum, clean_amplitude_spectrum = self.convert_to_stft(clean_wav)
                    noisy_complex_spectrum, noisy_amplitude_spectrum = self.convert_to_stft(noisy_wav)

                    stft_batch_clean.append((clean_complex_spectrum, clean_amplitude_spectrum))
                    stft_batch_noisy.append((noisy_complex_spectrum, noisy_amplitude_spectrum))

                yield stft_batch_clean, stft_batch_noisy

        # Yield orice date rămase (dacă sunt mai puține decât batch_size)
        if buffer:
            stft_batch_clean = []
            stft_batch_noisy = []
            for sample in buffer:
                clean_wav = torch.tensor(sample['clean']['array'], dtype=torch.float32).unsqueeze(0)
                noisy_wav = torch.tensor(sample['noisy']['array'], dtype=torch.float32).unsqueeze(0)

                clean_complex_spectrum, clean_amplitude_spectrum = self.convert_to_stft(clean_wav)
                noisy_complex_spectrum, noisy_amplitude_spectrum = self.convert_to_stft(noisy_wav)

                stft_batch_clean.append((torch.tensor(clean_complex_spectrum, dtype=torch.float32),
                                         torch.tensor(clean_amplitude_spectrum, dtype=torch.float32)))
                stft_batch_noisy.append((torch.tensor(noisy_complex_spectrum, dtype=torch.float32),
                                         torch.tensor(noisy_amplitude_spectrum, dtype=torch.float32)))

            yield stft_batch_clean, stft_batch_noisy

    def convert_to_stft(self, in_wav):
        complex_spectrum = torch.stft(in_wav, n_fft=self.configs.n_fft, hop_length=self.configs.hop_length,
                                      window=torch.hamming_window(self.configs.n_fft), return_complex=True)  # (B, F, T)
        amplitude_spectrum = torch.abs(complex_spectrum)

        # batch = self.configs.batch_size
        batch = 1
        # Preprocesarea spectrului
        complex_spectrum = torch.view_as_real(complex_spectrum)  # (B, F, T, 2)
        complex_spectrum = torch.permute(complex_spectrum, dims=(0, 2, 3, 1))
        _, frames, channels, frequency = complex_spectrum.shape
        complex_spectrum = torch.reshape(complex_spectrum, shape=(batch, frames, channels, frequency))
        amplitude_spectrum = torch.permute(amplitude_spectrum, dims=(0, 2, 1))
        amplitude_spectrum = torch.reshape(amplitude_spectrum, shape=(batch, frames, 1, frequency))

        return complex_spectrum, amplitude_spectrum


if __name__ == "__main__":
    # Salvarea configurației de antrenare
    configs = TrainConfig()
    with open("config.json", mode="w", encoding="utf-8") as file:
        json.dump(configs.__dict__, file, indent=4)
    ds = AudioDataset(configs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSubPathExtension(configs).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), configs.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_test_acc = 0.0

    for epoch in range(configs.epochs):
        print(f"Epoch {epoch + 1}/{configs.epochs}")

        # Antrenare
        model.train()
        total_loss = 0
        batch_count = 0

        for stft_batch_clean, stft_batch_noisy in ds.train_batches:
            for (clean_complex_spectrum, clean_amplitude_spectrum), (
                    noisy_complex_spectrum, noisy_amplitude_spectrum) in zip(stft_batch_clean, stft_batch_noisy):
                noisy_complex_spectrum = noisy_complex_spectrum.to(device)
                noisy_amplitude_spectrum = noisy_amplitude_spectrum.to(device)
                clean_amplitude_spectrum = clean_amplitude_spectrum.to(device)

                # Inițializarea stărilor ascunse ale rețelei RNN
                batch = configs.batch_size
                groups = configs.dual_path_extension["parameters"]["groups"]
                inter_hidden_size = configs.dual_path_extension["parameters"]["inter_hidden_size"]
                num_modules = configs.dual_path_extension["num_modules"]
                num_bands = sum(configs.bands_num_in_groups)
                in_hidden_state = [
                    [torch.zeros(1, batch * num_bands, inter_hidden_size // groups, device=device) for _ in
                     range(groups)]
                    for _ in range(num_modules)]

                predicted_clean_spectrum, _ = model(noisy_complex_spectrum, noisy_amplitude_spectrum, in_hidden_state)
                loss = criterion(predicted_clean_spectrum, clean_amplitude_spectrum)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        scheduler.step(avg_train_loss)

        # Evaluare
        model.eval()
        test_loss = 0
        test_batch_count = 0

        with torch.no_grad():
            for stft_batch_clean, stft_batch_noisy in ds.test_batches:
                (clean_complex_spectrum, clean_amplitude_spectrum) = stft_batch_clean[0]
                (noisy_complex_spectrum, noisy_amplitude_spectrum) = stft_batch_noisy[0]

                clean_amplitude_spectrum = clean_amplitude_spectrum.to(device)
                noisy_complex_spectrum = noisy_complex_spectrum.to(device)
                noisy_amplitude_spectrum = noisy_amplitude_spectrum.to(device)

                batch = configs.batch_size
                groups = configs.dual_path_extension["parameters"]["groups"]
                inter_hidden_size = configs.dual_path_extension["parameters"]["inter_hidden_size"]
                num_modules = configs.dual_path_extension["num_modules"]
                num_bands = sum(configs.bands_num_in_groups)
                in_hidden_state = [
                    [torch.zeros(1, batch * num_bands, inter_hidden_size // groups, device=device) for _ in
                     range(groups)]
                    for _ in range(num_modules)]

                predicted_clean_spectrum, _ = model(noisy_complex_spectrum, noisy_amplitude_spectrum, in_hidden_state)
                loss = criterion(predicted_clean_spectrum, clean_amplitude_spectrum)

                test_loss += loss.item()
                test_batch_count += 1

        avg_test_loss = test_loss / test_batch_count if test_batch_count > 0 else 0
        print(f"Test Loss: {avg_test_loss:.4f}")

        # Salvăm modelul dacă este cel mai bun
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), "best_fspen_model.pth")
            print("Model improved, saved!")

    # first_train_batch = next(ds.train_batches)
    # first_sample_complex, first_sample_amplitude = first_train_batch[0]
    #
    # batch = 1
    # # Inițializarea stărilor ascunse ale rețelei RNN
    # groups = configs.dual_path_extension["parameters"]["groups"]
    # inter_hidden_size = configs.dual_path_extension["parameters"]["inter_hidden_size"]
    # num_modules = configs.dual_path_extension["num_modules"]
    # num_bands = sum(configs.bands_num_in_groups)
    # in_hidden_state = [[torch.zeros(1, batch * num_bands, inter_hidden_size // groups) for _ in range(groups)]
    #                    for _ in range(num_modules)]
    #
    # # Profilarea modelului
    # flops, params = profile(model, inputs=(first_sample_complex, first_sample_amplitude, in_hidden_state))
    # flops, params = clever_format(nums=[flops, params], format="%0.4f")
    # print(f"flops: {flops} \nparams: {params}")