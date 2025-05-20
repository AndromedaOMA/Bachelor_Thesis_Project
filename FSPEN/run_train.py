import json
import torch
from thop import profile, clever_format
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.train_configs import TrainConfig
from models.fspen import FullSubPathExtension

from data.voicebank_demand_16K import VoiceBankDEMAND


"""
Loss: 0.051752, Train MSE: 0.040924
Validation MSE: 0.0409
Model improved. Saved.
==============================
Train MSE: 0.04092437628047548
"""

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


def train(model, data_loader, loss_fn, optimizer, device, epochs, configs):
    patience = 10
    best_mse = float("inf")
    no_improve_epochs = 0

    for epoch in range(epochs):
        print(f"=== Epoch: {epoch + 1}/{epochs} ===")
        train_per_epoch(model, data_loader, loss_fn, optimizer, device)

        mse = check_mse(data_loader, model, loss_fn, device, configs)
        print(f"Validation MSE: {mse:.4f}")

        if mse < best_mse:
            best_mse = mse
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model improved. Saved.")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        print("===" * 10)


def train_per_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()

    for batch in tqdm(data_loader):
        complex_input = batch["noisy_complex"].to(device).float().squeeze(1)
        amplitude_input = batch["noisy_amplitude"].to(device).float().squeeze(1)
        target = batch["clean_amplitude"].to(device).float().squeeze(1)

        # print("complex_input shape:", complex_input.shape)
        # print("amplitude_input shape:", amplitude_input.shape)
        # print("target shape:", target.shape)

        batch_size = complex_input.shape[0]
        hidden_state = prepare_initial_hidden_state(
            batch=batch_size,
            num_bands=sum(configs.bands_num_in_groups),
            num_modules=configs.dual_path_extension["num_modules"],
            groups=configs.dual_path_extension["parameters"]["groups"],
            inter_hidden_size=configs.dual_path_extension["parameters"]["inter_hidden_size"],
            device=device
        )

        prediction = model(complex_input, amplitude_input, hidden_state)
        predicted_amplitude = prediction[0]
        loss = loss_fn(predicted_amplitude, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_mse = check_mse(data_loader, model, loss_fn, device, configs)

    print(f"Loss: {loss.item():.6f}, Train MSE: {train_mse:.6f}")


# def check_acc(data_loader, model):
#     no_correct = 0
#     no_samples = 0
#     model.eval()
#
#     with torch.no_grad():
#         for input, target in data_loader:
#             input, target = input.to(device), target.to(device)
#
#             prediction = model(input)
#             predicted_labels = prediction.argmax(dim=1)
#
#             no_correct += (predicted_labels == target).sum().item()
#             no_samples += target.size(0)
#
#     train_acc = no_correct / no_samples
#     model.train()
#     return train_acc


def check_mse(data_loader, model, loss_fn, device, configs):
    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            complex_input = batch["noisy_complex"].to(device).float().squeeze(1)
            # print("complex_input shape before squeeze:", complex_input.shape)
            amplitude_input = batch["noisy_amplitude"].to(device).float().squeeze(1)
            target = batch["clean_amplitude"].to(device).float().squeeze(1)

            batch_size = complex_input.shape[0]
            hidden_state = prepare_initial_hidden_state(
                batch=batch_size,
                num_bands=sum(configs.bands_num_in_groups),
                num_modules=configs.dual_path_extension["num_modules"],
                groups=configs.dual_path_extension["parameters"]["groups"],
                inter_hidden_size=configs.dual_path_extension["parameters"]["inter_hidden_size"],
                device=device
            )

            prediction = model(complex_input, amplitude_input, hidden_state)
            predicted_amplitude = prediction[0]
            loss = loss_fn(predicted_amplitude, target)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
    model.train()
    return total_loss / total_samples


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    configs = TrainConfig()
    with open("config.json", mode="w", encoding="utf-8") as file:
        json.dump(configs.__dict__, file, indent=4)

    model = FullSubPathExtension(configs).to(device)

    dataset = VoiceBankDEMAND(device, configs)
    print(f"dataset has {len(dataset)} samples!")
    sample = dataset[0]
    print(f"noisy_amplitude.shape: {sample['noisy_amplitude'].shape}")
    print(f"noisy_complex.shape: {sample['noisy_complex'].shape}")
    print(f"clean_amplitude.shape: {sample['clean_amplitude'].shape}")
    print(f"clean_complex.shape: {sample['clean_complex'].shape}")

    print("=======" * 10)

    complex_spectrum = sample["noisy_complex"].to(device=device, dtype=torch.float32)       # (B=1, T, 2, F)
    amplitude_spectrum = sample["noisy_amplitude"].to(device=device, dtype=torch.float32)   # (B=1, T, 1, F)
    in_hidden_state = prepare_initial_hidden_state(
        batch=complex_spectrum.shape[0],
        num_bands=sum(configs.bands_num_in_groups),
        num_modules=configs.dual_path_extension["num_modules"],
        groups=configs.dual_path_extension["parameters"]["groups"],
        inter_hidden_size=configs.dual_path_extension["parameters"]["inter_hidden_size"],
        device=device
    )

    flops, params = profile(model, inputs=(complex_spectrum, amplitude_spectrum, in_hidden_state))
    flops, params = clever_format([flops, params], format="%0.4f")
    print(f"FLOPs: {flops}\nParams: {params}")

    print("=======" * 10)

    train_subset = dataset.train_data
    test_subset = dataset.test_data

    train_dataset = torch.utils.data.Subset(dataset, range(len(train_subset)))
    test_dataset = torch.utils.data.Subset(dataset, range(len(train_subset), len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size)

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train(model, train_loader, loss_fn, optimiser, device, configs.epochs, configs)

    mse = check_mse(train_loader, model, loss_fn, device, configs)
    print(f"Train MSE: {mse}")
    torch.save(model.state_dict(), "FSPEN.pth")
