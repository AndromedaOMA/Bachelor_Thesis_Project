import os
import pandas as pd

from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

from scipy.io import wavfile

from FSPEN.configs.train_configs import TrainConfig

if __name__ == "__main__":
    configs = TrainConfig()

    metrics = []
    out_dir = 'outputs/2sec_samples'
    clean_dir = f'{out_dir}/clean/'
    enhanced_dir = f'{out_dir}/enhanced/'
    noisy_dir = f'{out_dir}/noisy/'
    suffixes = [file_name.replace('clean_', '') for file_name in os.listdir(clean_dir)]

    for suffix in tqdm(suffixes):
        clean_path = clean_dir + f'clean_{suffix}'
        enhanced_path = enhanced_dir + f'enhanced_{suffix}'
        noisy_path = noisy_dir + f'noisy_{suffix}'

        sr_c, clean = wavfile.read(clean_path)
        sr_n, noisy = wavfile.read(noisy_path)
        sr_e, enhanced = wavfile.read(enhanced_path)

        """PESQ"""
        pesq_noisy = pesq(configs.sample_rate, clean, noisy, 'wb')
        pesq_enhanced = pesq(configs.sample_rate, clean, enhanced, 'wb')
        """STOI"""
        stoi_noisy = stoi(clean, noisy, configs.sample_rate, extended=False)
        stoi_enhanced = stoi(clean, enhanced, configs.sample_rate, extended=False)

        metrics.append({
            'file_suffix': suffix,
            'pesq_noisy': pesq_noisy,
            'pesq_enhanced': pesq_enhanced,
            'stoi_noisy': stoi_noisy,
            'stoi_enhanced': stoi_enhanced
        })

    df = pd.DataFrame(metrics)
    output_csv = 'metrics.csv'
    df.to_csv(output_csv, index=False)
