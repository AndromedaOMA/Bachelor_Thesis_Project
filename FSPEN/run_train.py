import json
import torch
from thop import profile, clever_format
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_pesq import PesqLoss

from configs.train_configs import TrainConfig
from models.efspen import FullSubPathExtension

from data.voicebank_demand_16K import VoiceBankDEMAND

""" best_model_0.0502 -> 4s (WORSE! Too much padding resulted in the human voice being thickened)
Loss: 0.0736, Train MSE: 0.0846, Test MSE: 0.0502, Current learning rate: 0.0001
Model improved. Saved. Current Train MSE: 0.0846. Current Test MSE: 0.0502
"""

""" best_model_0.0613 -> 2 s (+ MultiheadAttention) [Trained for all 100 epochs!]
Loss: 0.0915, Train MSE: 0.0900, Test MSE: 0.0613, Current learning rate: 0.0001
Model improved. Saved. Current Train MSE: 0.0900. Current Test MSE: 0.0613
"""

""" best_model_0.0676 -> 2 s (Ideal beacause its samples does not have too much paddings)
Loss: 0.1345, Train MSE: 0.1002, Test MSE: 0.0676, Current learning rate: 0.0000
Model improved. Saved. Current Train MSE: 0.1002. Current Test MSE: 0.0676
==============================
Early stopping at epoch 63. Best Test MSE: 0.0676
"""

""" best_model_0.0140 -> 1 s
Loss: 0.0207, Train MSE: 0.0304, Test MSE: 0.0140, Current learning rate: 0.0021
Model improved. Saved. Current Train MSE: 0.0304. Current Test MSE: 0.0140
==============================
Early stopping at epoch 34. Best Test MSE: 0.0140
"""

""" best_model_0.0135 -> 1 s
Loss: 0.0313, Train MSE: 0.0296, Test MSE: 0.0135, Current learning rate: 0.0006
Model improved. Saved. Current Train MSE: 0.0296. Current Test MSE: 0.0135
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


def train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, epochs, configs):
    patience = 10
    best_test_mse = float("inf")
    no_improve_epochs = 0

    for epoch in range(epochs):
        print(f"=== Epoch: {epoch + 1}/{epochs} ===")
        train_mse, test_mse = train_per_epoch(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            no_improve_epochs = 0
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"Model improved. Saved. "
                  f"Current Train MSE: {train_mse:.4f}. "
                  f"Current Test MSE: {test_mse:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best Test MSE: {best_test_mse:.4f}")
                break
        print("===" * 10)


def train_per_epoch(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device):
    model.train()
    torch.autograd.set_detect_anomaly(True)

    for batch in tqdm(train_loader):
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
        predicted_complex = prediction[0]  # shape: [B, T, 2, F]
        predicted_amplitude = torch.sqrt(
            predicted_complex[:, :, 0, :] ** 2 + predicted_complex[:, :, 1, :] ** 2 + 1e-8  # Re-compute the predicted_amplitude
        ).unsqueeze(2)  # shape: [B, T, 1, F]
        loss = loss_fn(predicted_amplitude, target)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

    train_mse = check_mse(train_loader, model, loss_fn, device, configs)
    test_mse = check_mse(test_loader, model, loss_fn, device, configs)
    current_lr = scheduler.get_last_lr()[0]

    print(f"Loss: {loss.item():.4f}, "
          f"Train MSE: {train_mse:.4f}, "
          f"Test MSE: {test_mse:.4f}, "
          f"Current learning rate: {current_lr:.4f}")

    return train_mse, test_mse


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
            predicted_complex = prediction[0]  # shape: [B, T, 2, F]
            predicted_amplitude = torch.sqrt(
                predicted_complex[:, :, 0, :] ** 2 + predicted_complex[:, :, 1, :] ** 2 + 1e-8  # Re-compute the predicted_amplitude
            ).unsqueeze(2)  # shape: [B, T, 1, F]
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
    sample = dataset[0]
    # print(f"noisy_amplitude.shape: {sample['noisy_amplitude'].shape}")
    # print(f"noisy_complex.shape: {sample['noisy_complex'].shape}")
    # print(f"clean_amplitude.shape: {sample['clean_amplitude'].shape}")
    # print(f"clean_complex.shape: {sample['clean_complex'].shape}")
    # print("=======" * 10)

    # complex_spectrum = sample["noisy_complex"].to(device=device, dtype=torch.float32)       # (B=1, T, 2, F)
    # amplitude_spectrum = sample["noisy_amplitude"].to(device=device, dtype=torch.float32)   # (B=1, T, 1, F)
    # in_hidden_state = prepare_initial_hidden_state(
    #     batch=complex_spectrum.shape[0],
    #     num_bands=sum(configs.bands_num_in_groups),
    #     num_modules=configs.dual_path_extension["num_modules"],
    #     groups=configs.dual_path_extension["parameters"]["groups"],
    #     inter_hidden_size=configs.dual_path_extension["parameters"]["inter_hidden_size"],
    #     device=device
    # )

    # # FullSubPathExtension profile
    # flops, params = profile(model, inputs=(complex_spectrum, amplitude_spectrum, in_hidden_state))
    # flops, params = clever_format([flops, params], format="%0.4f")
    # print(f"Original FSPEN Model\nFLOPs: {flops}\nParams: {params}")

    print("=======" * 10)

    train_dataset = VoiceBankDEMAND(device, configs, mode="train")
    test_dataset = VoiceBankDEMAND(device, configs, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size)

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=configs.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=2)

    train(model, train_loader, test_loader, loss_fn, optimiser, scheduler, device, configs.epochs, configs)

"""Librispeech"""
#
# import os
# import json
# import torch
# import itertools
# import pandas as pd
# import gc
# import time
# from datetime import datetime
# from torch import nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
#
# from configs.train_configs import TrainConfig
# from models.efspen import FullSubPathExtension
# from data.voicebank_demand_16K import VoiceBankDEMAND
#
# # Optimize memory segments for 8GB GPU
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#
#
# # --- HELPER FUNCTIONS ---
#
# def prepare_initial_hidden_state(batch, num_bands, num_modules, groups, inter_hidden_size, device):
#     return [
#         [torch.zeros(1, batch * num_bands, inter_hidden_size // groups).to(device=device, dtype=torch.float32)
#          for _ in range(groups)]
#         for _ in range(num_modules)
#     ]
#
#
# def check_mse(data_loader, model, loss_fn, device, configs):
#     total_loss, total_samples = 0.0, 0
#     model.eval()
#     with torch.no_grad():
#         for batch in data_loader:
#             complex_input = batch["noisy_complex"].to(device).float().squeeze(1)
#             amplitude_input = batch["noisy_amplitude"].to(device).float().squeeze(1)
#             target = batch["clean_amplitude"].to(device).float().squeeze(1)
#
#             hidden_state = prepare_initial_hidden_state(
#                 batch=complex_input.shape[0],
#                 num_bands=sum(configs.bands_num_in_groups),
#                 num_modules=configs.dual_path_extension["num_modules"],
#                 groups=configs.dual_path_extension["parameters"]["groups"],
#                 inter_hidden_size=configs.dual_path_extension["parameters"]["inter_hidden_size"],
#                 device=device
#             )
#
#             prediction = model(complex_input, amplitude_input, hidden_state)
#             predicted_complex = prediction[0]
#             predicted_amplitude = torch.sqrt(
#                 predicted_complex[:, :, 0, :] ** 2 + predicted_complex[:, :, 1, :] ** 2 + 1e-8).unsqueeze(2)
#             loss = loss_fn(predicted_amplitude, target)
#             total_loss += loss.item() * complex_input.shape[0]
#             total_samples += complex_input.shape[0]
#     return total_loss / total_samples
#
#
# def train_per_epoch(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, configs):
#     model.train()
#     total_train_loss = 0
#     for batch in tqdm(train_loader, desc="Training", leave=False):
#         complex_input = batch["noisy_complex"].to(device).float().squeeze(1)
#         amplitude_input = batch["noisy_amplitude"].to(device).float().squeeze(1)
#         target = batch["clean_amplitude"].to(device).float().squeeze(1)
#
#         hidden_state = prepare_initial_hidden_state(
#             batch=complex_input.shape[0],
#             num_bands=sum(configs.bands_num_in_groups),
#             num_modules=configs.dual_path_extension["num_modules"],
#             groups=configs.dual_path_extension["parameters"]["groups"],
#             inter_hidden_size=configs.dual_path_extension["parameters"]["inter_hidden_size"],
#             device=device
#         )
#
#         prediction = model(complex_input, amplitude_input, hidden_state)
#         pred_complex = prediction[0]
#         pred_amplitude = torch.sqrt(pred_complex[:, :, 0, :] ** 2 + pred_complex[:, :, 1, :] ** 2 + 1e-8).unsqueeze(2)
#
#         loss = loss_fn(pred_amplitude, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         total_train_loss += loss.item()
#
#     avg_train_loss = total_train_loss / len(train_loader)
#     train_mse = check_mse(train_loader, model, loss_fn, device, configs)
#     test_mse = check_mse(test_loader, model, loss_fn, device, configs)
#     return train_mse, test_mse, avg_train_loss
#
#
# # --- MAIN GRID SEARCH ---
#
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     save_at = "./checkpoints/grid_search/"
#     os.makedirs(save_at, exist_ok=True)
#
#     # 1. Configuration Constants
#     PATIENCE = 10
#     TOTAL_EPOCHS = 100
#
#     search_space = {
#         "intra_hidden": [16, 32],
#         "groups": [4, 8],
#         "modules": [2, 3]
#     }
#
#     keys, values = zip(*search_space.items())
#     combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
#     results_log = []
#
#     for idx, params in enumerate(combinations):
#         start_run_time = time.time()
#         run_name = f"RUN{idx + 1}_h{params['intra_hidden']}_g{params['groups']}_m{params['modules']}"
#         print(f"\n{'=' * 10} {run_name} {'=' * 10}")
#
#         try:
#             configs = TrainConfig()
#             configs.dual_path_extension["parameters"]["intra_hidden_size"] = params["intra_hidden"]
#             configs.dual_path_extension["parameters"]["groups"] = params["groups"]
#             configs.dual_path_extension["num_modules"] = params["modules"]
#             configs.batch_size = 16
#
#             train_loader = DataLoader(VoiceBankDEMAND(device, configs, mode="train"), batch_size=configs.batch_size,
#                                       shuffle=True)
#             test_loader = DataLoader(VoiceBankDEMAND(device, configs, mode="test"), batch_size=configs.batch_size)
#
#             model = FullSubPathExtension(configs).to(device)
#             loss_fn = nn.MSELoss()
#             optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
#
#             best_val_loss = float('inf')
#             no_improvements = 0
#             history = {"avg_train_loss": [], "avg_val_loss": [], "train_mse": [], "test_mse": []}
#
#             # 2. Training Loop with Early Stopping
#             for epoch in range(TOTAL_EPOCHS):
#                 train_mse, test_mse, avg_train_loss = train_per_epoch(model, train_loader, test_loader, loss_fn,
#                                                                       optimizer, scheduler, device, configs)
#
#                 history["avg_train_loss"].append(avg_train_loss)
#                 history["avg_val_loss"].append(test_mse)
#                 history["train_mse"].append(train_mse)
#                 history["test_mse"].append(test_mse)
#
#                 print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")
#
#                 # 3. Check for Improvement
#                 if test_mse < best_val_loss - 1e-5:
#                     no_improvements = 0
#                     best_val_loss = test_mse
#                     timestamp = datetime.now().strftime("%H-%M-%S")
#
#                     filename = f"{run_name}_ep{epoch + 1}_mse{best_val_loss:.4f}_{timestamp}.pth"
#                     checkpoint_path = os.path.join(save_at, filename)
#
#                     stored_data = {
#                         'epoch': epoch + 1,
#                         'params': params,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'scheduler_state_dict': scheduler.state_dict(),
#                         'mse_loss': best_val_loss,
#                         'history': history,
#                         'total_run_time': time.time() - start_run_time
#                     }
#                     torch.save(stored_data, checkpoint_path)
#                     torch.save(model.state_dict(), os.path.join(save_at, f"{run_name}_BEST.pth"))
#                     print(f"--> Saved improved checkpoint: {filename}")
#                 else:
#                     no_improvements += 1
#                     print(f"No improvement for {no_improvements} epoch(s).")
#
#                 # 4. Trigger Early Stopping
#                 if no_improvements >= PATIENCE:
#                     print(f"Early stopping triggered for {run_name} at epoch {epoch + 1}.")
#                     break
#
#             results_log.append({**params, "best_test_mse": best_val_loss, "epochs_run": epoch + 1})
#
#         except torch.OutOfMemoryError:
#             print(f"!!! OOM Error on {run_name} !!!")
#             results_log.append({**params, "best_test_mse": "OOM", "epochs_run": 0})
#         finally:
#             if 'model' in locals(): model.cpu()
#             del model, optimizer, scheduler, train_loader, test_loader
#             gc.collect()
#             torch.cuda.empty_cache()
#
#     pd.DataFrame(results_log).to_csv("grid_search_summary_with_early_stopping.csv", index=False)
#     print("\nGrid Search Complete.")