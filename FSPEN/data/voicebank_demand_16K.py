import torch
import io
import torchaudio
from datasets import load_dataset, Audio
from torch.utils.data import Dataset

from FSPEN.configs.train_configs import TrainConfig


class VoiceBankDEMAND(Dataset):
    def __init__(self, device, configs: TrainConfig, mode='train'):
        self.device = device
        self.configs = configs
        self.ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
        self.ds = self.ds.cast_column("noisy", Audio(decode=False))
        self.ds = self.ds.cast_column("clean", Audio(decode=False))
        self.sample_rate = 16000
        self.num_samples = configs.sample_length
        self.train_data = self.ds["train"]
        self.test_data = self.ds["test"]
        self.mode = mode

    def __len__(self):
        return len(self.train_data) if self.mode == 'train' else len(self.test_data)

    def __getitem__(self, index):
        data = self.train_data if self.mode == 'train' else self.test_data

        # 2. Extract raw bytes from the dictionary
        id_dict = data[index]['id']
        noisy_dict = data[index]['noisy']
        clean_dict = data[index]['clean']

        # 3. Use torchaudio to load from raw bytes (bypasses torchcodec)
        # noisy_dict['bytes'] contains the raw WAV file data
        noisy_waveform, sr_n = torchaudio.load(io.BytesIO(noisy_dict['bytes']))
        clean_waveform, sr_c = torchaudio.load(io.BytesIO(clean_dict['bytes']))

        # 4. Resample if Hugging Face version differs (though this dataset is 16k)
        if sr_n != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_n, self.sample_rate).to(noisy_waveform.device)
            noisy_waveform = resampler(noisy_waveform)
        if sr_c != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_c, self.sample_rate).to(clean_waveform.device)
            clean_waveform = resampler(clean_waveform)

        # 5. Rest of your processing logic
        noisy_waveform = self._process_waveform(noisy_waveform)
        clean_waveform = self._process_waveform(clean_waveform)

        # Move to GPU
        noisy_waveform = noisy_waveform.to(self.device)
        clean_waveform = clean_waveform.to(self.device)

        noisy_complex, noisy_amplitude = self._prepare_spectrum_inputs(noisy_waveform)
        clean_complex, clean_amplitude = self._prepare_spectrum_inputs(clean_waveform)

        return {
            'id': id_dict,
            'noisy_waveform': noisy_waveform,
            'clean_waveform': clean_waveform,
            'noisy_complex': noisy_complex,
            'noisy_amplitude': noisy_amplitude,
            'clean_complex': clean_complex,
            'clean_amplitude': clean_amplitude
        }

    def _process_waveform(self, signal):
        signal = signal.to(self.device)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            signal = torch.nn.functional.pad(signal, (0, self.num_samples - signal.shape[1]))
        return signal

    def _prepare_spectrum_inputs(self, waveforms: torch.Tensor):
        """Complex spectrum: (B, T, 2, F) | Amplitude spectrum: (B, T, 1, F)"""
        B = waveforms.size(0)

        complex_spectrum = torch.stft(
            waveforms,
            n_fft=self.configs.n_fft,
            hop_length=self.configs.hop_length,
            window=torch.hamming_window(self.configs.n_fft).to(self.device),
            return_complex=True,
        )

        amplitude_spectrum = torch.abs(complex_spectrum)
        amplitude_spectrum = amplitude_spectrum.permute(0, 2, 1).unsqueeze(2)
        amplitude_spectrum = (amplitude_spectrum - amplitude_spectrum.mean()) / (amplitude_spectrum.std() + 1e-5)

        complex_spectrum = torch.view_as_real(complex_spectrum)
        complex_spectrum = complex_spectrum.permute(0, 2, 3, 1)

        return complex_spectrum, amplitude_spectrum


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = TrainConfig()
    dataset = VoiceBankDEMAND(device, configs)

    sample = dataset[0]
    print(f"noisy_waveform.shape: {sample['noisy_waveform'].shape}")
    print(f"clean_waveform.shape: {sample['clean_waveform'].shape}")
    print(f"noisy_amplitude.shape: {sample['noisy_amplitude'].shape}")
    print(f"noisy_complex.shape: {sample['noisy_complex'].shape}")
    print(f"clean_amplitude.shape: {sample['clean_amplitude'].shape}")
    print(f"clean_complex.shape: {sample['clean_complex'].shape}")
