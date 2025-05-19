import json
import torch
from thop import profile, clever_format

from configs.train_configs import TrainConfig
from models.fspen import FullSubPathExtension


def prepare_spectrum_inputs(waveforms: torch.Tensor, configs: TrainConfig):
    """
    Compute STFT and format tensors for model input.
    Args:
        waveforms (torch.Tensor): Input waveform tensor of shape (B, T).
        configs (TrainConfig): Configuration object with STFT and model parameters.

    Returns:
        complex_spectrum (torch.Tensor): Complex spectrum tensor for the model.
        amplitude_spectrum (torch.Tensor): Amplitude spectrum tensor for the model.
    """
    batch_size = waveforms.size(0)

    # STFT returns (B, F, T) complex
    complex_spectrum = torch.stft(
        waveforms,
        n_fft=configs.n_fft,
        hop_length=configs.hop_length,
        window=torch.hamming_window(configs.n_fft),
        return_complex=True,
    )

    # Amplitude spectrum (B, F, T) → (B, T, 1, F)
    amplitude_spectrum = torch.abs(complex_spectrum)
    amplitude_spectrum = amplitude_spectrum.permute(0, 2, 1).unsqueeze(2)

    # Convert complex spectrum to real (B, F, T, 2)
    complex_spectrum = torch.view_as_real(complex_spectrum)
    # Rearrange to (B, T, 2, F)
    complex_spectrum = complex_spectrum.permute(0, 2, 3, 1)

    # Final reshape: (B, T, 2, F)
    return complex_spectrum, amplitude_spectrum


if __name__ == "__main__":
    configs = TrainConfig()
    with open("config.json", mode="w", encoding="utf-8") as file:
        json.dump(configs.__dict__, file, indent=4)

    model = FullSubPathExtension(configs)

    batch = 32
    waveforms = torch.randn(batch, configs.train_points)

    complex_spectrum, amplitude_spectrum = prepare_spectrum_inputs(waveforms, configs)

    groups = configs.dual_path_extension["parameters"]["groups"]
    inter_hidden_size = configs.dual_path_extension["parameters"]["inter_hidden_size"]
    num_modules = configs.dual_path_extension["num_modules"]
    num_bands = sum(configs.bands_num_in_groups)
    frames = complex_spectrum.shape[1]

    in_hidden_state = [
        [torch.zeros(1, batch * num_bands, inter_hidden_size // groups) for _ in range(groups)]
        for _ in range(num_modules)
    ]

    flops, params = profile(model, inputs=(complex_spectrum, amplitude_spectrum, in_hidden_state))
    flops, params = clever_format([flops, params], format="%0.4f")

    print(f"FLOPs: {flops}\nParams: {params}")
