import torch
import torchaudio
from FSPEN.configs.train_configs import TrainConfig
from FSPEN.models.efspen import FullSubPathExtension


class EFSPENEnhancer:
    def __init__(self, checkpoint_path, device='cuda:0'):
        # Fix: Explicit device indexing to prevent CUDA unpacking errors
        self.device = torch.device(device)
        self.configs = TrainConfig()
        self.model = FullSubPathExtension(self.configs).to(self.device)

        # Security: Use weights_only=True to resolve FutureWarning
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_hidden(self, batch_size):
        # Matches your dual-path RNN requirements
        return [
            [torch.zeros(1, batch_size * sum(self.configs.bands_num_in_groups),
                         self.configs.dual_path_extension["parameters"]["inter_hidden_size"] // 8).to(self.device)
             for _ in range(8)]
            for _ in range(self.configs.dual_path_extension["num_modules"])
        ]

    @torch.no_grad()
    def enhance_audio(self, waveform):
        """ waveform: [1, T] tensor at 16kHz """
        # 1. Spectral Analysis (STFT)
        stft = torch.stft(waveform, n_fft=512, hop_length=256,
                          window=torch.hann_window(512).to(self.device), return_complex=True)

        # 2. Reshape for Model: (B, T, C, F)
        mag = torch.abs(stft).permute(0, 2, 1).unsqueeze(2)
        real = torch.real(stft).permute(0, 2, 1).unsqueeze(2)
        imag = torch.imag(stft).permute(0, 2, 1).unsqueeze(2)
        complex_spec = torch.cat([real, imag], dim=2)

        # 3. Model Inference
        hidden = self._prepare_hidden(batch_size=1)
        enhanced_complex, _ = self.model(complex_spec, mag, hidden)

        # 4. Synthesis (ISTFT)
        res_real = enhanced_complex[0, :, 0, :].T
        res_imag = enhanced_complex[0, :, 1, :].T
        enhanced_stft = torch.complex(res_real, res_imag)

        enhanced_wav = torch.istft(enhanced_stft, n_fft=512, hop_length=256,
                                   window=torch.hann_window(512).to(self.device))
        return enhanced_wav.cpu().unsqueeze(0)
