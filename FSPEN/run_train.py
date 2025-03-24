import json

import torch
from thop import profile, clever_format

from configs.train_configs import TrainConfig
from models.fspen import FullSubPathExtension

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


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
        self.dataset = load_dataset("parquet", data_files=self.data_files, streaming=True)
        self.configs = configs

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, idx):
    #     # Încărcăm datele audio din setul de date
    #     sample = self.dataset[idx]
    #
    #     # Accesăm corect semnalele audio din dicționar
    #     noisy_wav = torch.tensor(sample['noisy'], dtype=torch.float32)  # Conversie în tensor
    #     clean_wav = torch.tensor(sample['clean'], dtype=torch.float32)  # Conversie în tensor
    #
    #     # Preprocesarea semnalului cu zgomot
    #     noisy_wav = noisy_wav.squeeze(0)  # Îndepărtează dimensiunea redundantă (dacă există)
    #     clean_wav = clean_wav.squeeze(0)  # Îndepărtează dimensiunea redundantă (dacă există)
    #
    #     # Transformarea semnalelor audio într-un spectru complex
    #     noisy_complex_spectrum = torch.stft(noisy_wav, n_fft=self.configs.n_fft, hop_length=self.configs.hop_length,
    #                                         window=torch.hamming_window(self.configs.n_fft), return_complex=True)
    #     clean_complex_spectrum = torch.stft(clean_wav, n_fft=self.configs.n_fft, hop_length=self.configs.hop_length,
    #                                         window=torch.hamming_window(self.configs.n_fft), return_complex=True)
    #
    #     # Calcularea amplitudinii
    #     noisy_amplitude_spectrum = torch.abs(noisy_complex_spectrum)
    #     clean_amplitude_spectrum = torch.abs(clean_complex_spectrum)
    #
    #     # Preprocesarea spectrului complex
    #     noisy_complex_spectrum = torch.view_as_real(noisy_complex_spectrum)  # (B, F, T, 2)
    #     clean_complex_spectrum = torch.view_as_real(clean_complex_spectrum)  # (B, F, T, 2)
    #
    #     noisy_complex_spectrum = torch.permute(noisy_complex_spectrum, dims=(0, 2, 3, 1))  # (B, T, F, 2)
    #     clean_complex_spectrum = torch.permute(clean_complex_spectrum, dims=(0, 2, 3, 1))  # (B, T, F, 2)
    #
    #     # Asigurăm că dimensiunea este corectă
    #     batch_size, frames, channels, frequency = noisy_complex_spectrum.shape
    #     noisy_complex_spectrum = torch.reshape(noisy_complex_spectrum, shape=(batch_size, frames, channels, frequency))
    #     clean_complex_spectrum = torch.reshape(clean_complex_spectrum, shape=(batch_size, frames, channels, frequency))
    #
    #     # Preprocesarea spectrului de amplitudine
    #     noisy_amplitude_spectrum = torch.permute(noisy_amplitude_spectrum, dims=(0, 2, 1))  # (B, F, T)
    #     clean_amplitude_spectrum = torch.permute(clean_amplitude_spectrum, dims=(0, 2, 1))  # (B, F, T)
    #
    #     noisy_amplitude_spectrum = torch.reshape(noisy_amplitude_spectrum, shape=(batch_size, frames, 1, frequency))
    #     clean_amplitude_spectrum = torch.reshape(clean_amplitude_spectrum, shape=(batch_size, frames, 1, frequency))
    #
    #     return noisy_complex_spectrum, clean_complex_spectrum, noisy_amplitude_spectrum, clean_amplitude_spectrum

    def batches_generator(self, data):
        """YIELD (continuous "return") the current batches of self.configs.batch_size elements each"""
        buffer = []

        for example in data.shuffle(buffer_size=5000):
            buffer.append(example)

            if len(buffer) == self.configs.batch_size:
                batch = buffer[:self.configs.batch_size]
                buffer = buffer[self.configs.batch_size:]
                yield batch

        # Yield any remaining data (if less than batch_size)
        if buffer:
            yield buffer

    def convert_to_stft(self, in_wav):
        # Generarea unui input pentru testare
        # in_wav = torch.tensor(train_set["clean"][1])
        in_wav = torch.randn(1, configs.train_points)
        complex_spectrum = torch.stft(in_wav, n_fft=configs.n_fft, hop_length=configs.hop_length,
                                      window=torch.hamming_window(configs.n_fft), return_complex=True)  # (B, F, T)
        amplitude_spectrum = torch.abs(complex_spectrum)

        batch = 1
        # Preprocesarea spectrului
        complex_spectrum = torch.view_as_real(complex_spectrum)  # (B, F, T, 2)
        complex_spectrum = torch.permute(complex_spectrum, dims=(0, 2, 3, 1))
        _, frames, channels, frequency = complex_spectrum.shape
        complex_spectrum = torch.reshape(complex_spectrum, shape=(batch, frames, channels, frequency))
        amplitude_spectrum = torch.permute(amplitude_spectrum, dims=(0, 2, 1))
        amplitude_spectrum = torch.reshape(amplitude_spectrum, shape=(batch, frames, 1, frequency))


if __name__ == "__main__":
    # Salvarea configurației de antrenare
    configs = TrainConfig()
    with open("config.json", mode="w", encoding="utf-8") as file:
        json.dump(configs.__dict__, file, indent=4)
    model = FullSubPathExtension(configs)

    ds = AudioDataset(configs)
    # print(ds.dataset)
    train_batches = ds.batches_generator(ds.dataset['train'])
    test_batches = ds.batches_generator(ds.dataset['test'])

    first_train_batch = next(train_batches)
    first_sample = first_train_batch[0]
    # print(first_sample['clean'])

    # for batch in train_batches:
    #     for sample in batch:
    #         ds.convert_to_stft(sample)
    #         print(sample)
    #         break

    # Generarea unui input pentru testare
    # in_wav = torch.randn(1, configs.train_points)

    # Introducerea unui input valid din VoiceBank_DEMAND_16k
    in_wav = torch.tensor(first_sample['clean']['array'], dtype=torch.float32)
    in_wav = in_wav.unsqueeze(0)
    # print(f"Dimension in_wav: {in_wav.shape}")
    complex_spectrum = torch.stft(in_wav, n_fft=configs.n_fft, hop_length=configs.hop_length,
                                  window=torch.hamming_window(configs.n_fft), return_complex=True)  # (B, F, T)
    amplitude_spectrum = torch.abs(complex_spectrum)

    batch = 1
    # Preprocesarea spectrului
    complex_spectrum = torch.view_as_real(complex_spectrum)  # (B, F, T, 2)
    complex_spectrum = torch.permute(complex_spectrum, dims=(0, 2, 3, 1))
    _, frames, channels, frequency = complex_spectrum.shape
    complex_spectrum = torch.reshape(complex_spectrum, shape=(batch, frames, channels, frequency))
    amplitude_spectrum = torch.permute(amplitude_spectrum, dims=(0, 2, 1))
    amplitude_spectrum = torch.reshape(amplitude_spectrum, shape=(batch, frames, 1, frequency))

    # Inițializarea stărilor ascunse ale rețelei RNN
    groups = configs.dual_path_extension["parameters"]["groups"]
    inter_hidden_size = configs.dual_path_extension["parameters"]["inter_hidden_size"]
    num_modules = configs.dual_path_extension["num_modules"]
    num_bands = sum(configs.bands_num_in_groups)
    in_hidden_state = [[torch.zeros(1, batch * num_bands, inter_hidden_size // groups) for _ in range(groups)]
                       for _ in range(num_modules)]

    # Profilarea modelului
    flops, params = profile(model, inputs=(complex_spectrum, amplitude_spectrum, in_hidden_state))
    flops, params = clever_format(nums=[flops, params], format="%0.4f")
    print(f"flops: {flops} \nparams: {params}")
