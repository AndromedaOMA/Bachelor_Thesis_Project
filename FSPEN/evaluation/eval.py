import torch
from torch.utils.data import DataLoader
import soundfile as sf

from FSPEN.configs.train_configs import TrainConfig
from FSPEN.data.voicebank_demand_16K import VoiceBankDEMAND
from FSPEN.models.fspen import FullSubPathExtension


def prepare_initial_hidden_state(
        batch: int,
        num_bands: int,
        num_modules: int,
        groups: int,
        inter_hidden_size: int,
        device: str
):
    return [
        [torch.zeros(1, batch * num_bands, inter_hidden_size // groups).to(device=device, dtype=torch.float32)
         for _ in range(groups)]
        for _ in range(num_modules)
    ]


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = TrainConfig()
    model = FullSubPathExtension(configs).to(device)
    model.eval()
    state_dict = torch.load("../best_model_0.0135.pth")
    filtered_state_dict = {k: v for k, v in state_dict.items() if
                           k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(state_dict)

    test_dataset = VoiceBankDEMAND(device, configs, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size)

    for i, batch in enumerate(test_loader):
        if i <= 4:
            with torch.no_grad():
                clean = batch["clean_waveform"]
                noisy = batch["noisy_waveform"]
                complex_input = batch["noisy_complex"].to(device).float().squeeze(1)
                amplitude_input = batch["noisy_amplitude"].to(device).float().squeeze(1)
                batch_size = complex_input.shape[0]
                hidden_state = prepare_initial_hidden_state(
                    batch=batch_size,
                    num_bands=sum(configs.bands_num_in_groups),
                    num_modules=configs.dual_path_extension["num_modules"],
                    groups=configs.dual_path_extension["parameters"]["groups"],
                    inter_hidden_size=configs.dual_path_extension["parameters"]["inter_hidden_size"],
                    device=device
                )
                enhanced, _ = model(complex_input, amplitude_input, hidden_state)
                complex_spec = torch.complex(enhanced[:, :, 0, :], enhanced[:, :, 1, :])
                complex_spec = complex_spec.permute(0, 2, 1)
                window = torch.hann_window(512).to(device)
                audio_recon = torch.istft(complex_spec,
                                          n_fft=512,
                                          # hop_length=128,
                                          hop_length=256,
                                          win_length=512,
                                          window=window,
                                          center=True,
                                          normalized=False,
                                          onesided=True,
                                          length=64000)
                clean_np = clean.squeeze(1).cpu().numpy()
                noisy_np = noisy.squeeze(1).cpu().numpy()
                audio_recon_np = audio_recon.cpu().numpy()
                print(f"clean shape: {clean.shape}")
                print(f"noisy shape: {noisy.shape}")
                print(f"audio_recon shape: {audio_recon.shape}")

            for b in range(batch_size):
                sf.write(f'outputs/enhanced/enhanced_{i}_{b}.wav', audio_recon_np[b], 16000)
                sf.write(f'outputs/clean/clean_{i}_{b}.wav', clean_np[b], 16000)
                sf.write(f'outputs/noisy/noisy_{i}_{b}.wav', noisy_np[b], 16000)