import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset

from FSPEN.configs.train_configs import TrainConfig


class VoiceBankDEMAND(Dataset):
    def __init__(self, device, configs: TrainConfig):
        self.device = device
        self.configs = configs
        self.ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
        self.sample_rate = 16000
        self.num_samples = configs.sample_length
        self.train_data = self.ds["train"]
        self.test_data = self.ds["test"]

    def __len__(self):
        return len(self.train_data) + len(self.test_data)

    def __getitem__(self, index):
        noisy = self.train_data[index]['noisy']
        clean = self.train_data[index]['clean']

        # Convert to tensor
        noisy_waveform = torch.tensor(noisy['array']).unsqueeze(0)
        clean_waveform = torch.tensor(clean['array']).unsqueeze(0)

        # Preprocess
        noisy_waveform = self._process_waveform(noisy_waveform)
        clean_waveform = self._process_waveform(clean_waveform)

        # (1, T) → (B=1, T)
        noisy_waveform = noisy_waveform.to(self.device).squeeze(0).unsqueeze(0)
        clean_waveform = clean_waveform.to(self.device).squeeze(0).unsqueeze(0)

        # Compute spectra
        noisy_complex, noisy_amplitude = self._prepare_spectrum_inputs(noisy_waveform)
        clean_complex, clean_amplitude = self._prepare_spectrum_inputs(clean_waveform)

        return {
            "noisy_waveform": noisy_waveform,
            "clean_waveform": clean_waveform,
            "noisy_amplitude": noisy_amplitude.clone().detach().float(),
            "noisy_complex": noisy_complex.clone().detach().float(),
            "clean_amplitude": clean_amplitude.clone().detach().float(),
            "clean_complex": clean_complex.clone().detach().float(),
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
        """
        Compute STFT and format tensors for model input.
        Args:
            waveforms (torch.Tensor): (B, T)

        Returns:
            complex_spectrum: (B, T, 2, F)
            amplitude_spectrum: (B, T, 1, F)
        """
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

        complex_spectrum = torch.view_as_real(complex_spectrum)
        complex_spectrum = complex_spectrum.permute(0, 2, 3, 1)

        return complex_spectrum, amplitude_spectrum


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = TrainConfig()
    dataset = VoiceBankDEMAND(device, configs)
    print(f"dataset has {len(dataset)} samples!")

    sample = dataset[0]
    print(f"noisy_amplitude.shape: {sample['noisy_amplitude'].shape}")
    print(f"noisy_complex.shape: {sample['noisy_complex'].shape}")
    print(f"clean_amplitude.shape: {sample['clean_amplitude'].shape}")
    print(f"clean_complex.shape: {sample['clean_complex'].shape}")
