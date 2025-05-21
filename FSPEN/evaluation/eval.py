import torch
from torch.utils.data import DataLoader
import soundfile as sf

from FSPEN.configs.train_configs import TrainConfig
from FSPEN.data.voicebank_demand_16K import VoiceBankDEMAND
from FSPEN.models.fspen import FullSubPathExtension

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = TrainConfig()
    model = FullSubPathExtension(configs).to(device)
    model.eval()
    state_dict = torch.load("../best_model_0.0140.pth")
    # Filtrăm cheile incompatibile
    filtered_state_dict = {k: v for k, v in state_dict.items() if
                           k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(state_dict)

    test_dataset = VoiceBankDEMAND(device, configs, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size)

    for i, (noisy, clean) in enumerate(test_loader):
        with torch.no_grad():
            enhanced, _ = model(noisy)

        # Save files
        sf.write(f'outputs/enhanced/enhanced_{i}.wav', enhanced.squeeze().cpu().numpy(), 16000)
        sf.write(f'outputs/clean/clean_{i}.wav', clean.squeeze().cpu().numpy(), 16000)
        sf.write(f'outputs/noisy/noisy_{i}.wav', noisy.squeeze().cpu().numpy(), 16000)


