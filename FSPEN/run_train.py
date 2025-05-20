import json
import torch
from thop import profile, clever_format

from configs.train_configs import TrainConfig
from models.fspen import FullSubPathExtension

from data.voicebank_demand_16K import VoiceBankDEMAND

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    configs = TrainConfig()
    with open("config.json", mode="w", encoding="utf-8") as file:
        json.dump(configs.__dict__, file, indent=4)

    model = FullSubPathExtension(configs).to(device)
    model.eval()

    dataset = VoiceBankDEMAND(device, configs)
    print(f"dataset has {len(dataset)} samples!")

    sample = dataset[0]
    complex_spectrum = sample["noisy_complex"].to(device=device, dtype=torch.float32)       # (B=1, T, 2, F)
    amplitude_spectrum = sample["noisy_amplitude"].to(device=device, dtype=torch.float32)   # (B=1, T, 1, F)

    batch = complex_spectrum.shape[0]
    groups = configs.dual_path_extension["parameters"]["groups"]
    inter_hidden_size = configs.dual_path_extension["parameters"]["inter_hidden_size"]
    num_modules = configs.dual_path_extension["num_modules"]
    num_bands = sum(configs.bands_num_in_groups)
    frames = complex_spectrum.shape[1]

    in_hidden_state = [
        [torch.zeros(1, batch * num_bands, inter_hidden_size // groups).to(device=device, dtype=torch.float32)
         for _ in range(groups)]
        for _ in range(num_modules)
    ]

    flops, params = profile(model, inputs=(complex_spectrum, amplitude_spectrum, in_hidden_state))
    flops, params = clever_format([flops, params], format="%0.4f")

    print(f"FLOPs: {flops}\nParams: {params}")
