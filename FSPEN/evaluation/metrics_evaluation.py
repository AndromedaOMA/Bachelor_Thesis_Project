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
    out_dir = 'outputs/4sec_samples_mha/plots'
    clean_dir = f'{out_dir}/clean/'
    enhanced_dir = f'{out_dir}/enhanced/'
    noisy_dir = f'{out_dir}/noisy/'
    suffixes = [file_name.replace('clean_', '') for file_name in os.listdir(clean_dir)]

    pesq_gain_sum = 0
    stoi_gain_sum = 0

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
        """PESQ_gain"""
        pesq_gain = pesq_enhanced - pesq_noisy
        """PESQ_mean"""
        pesq_gain_sum += pesq_gain

        """STOI"""
        stoi_noisy = stoi(clean, noisy, configs.sample_rate, extended=False)
        stoi_enhanced = stoi(clean, enhanced, configs.sample_rate, extended=False)
        """STOI_gain"""
        stoi_gain = stoi_enhanced - stoi_noisy
        """STOI_mean"""
        stoi_gain_sum += stoi_gain

        metrics.append({
            'file_suffix': suffix,
            'pesq_noisy': pesq_noisy,
            'pesq_enhanced': pesq_enhanced,
            'pesq_gain': pesq_gain,
            'stoi_noisy': stoi_noisy,
            'stoi_enhanced': stoi_enhanced,
            'stoi_gain': stoi_gain
        })

    pesq_mean = pesq_gain_sum/len(metrics)
    stoi_mean = stoi_gain_sum/len(metrics)
    metrics.append({
        'pesq_mean': pesq_mean,
        'stoi_mean': stoi_mean
    })

    df = pd.DataFrame(metrics)
    output_csv = f'{out_dir}/metrics.csv'
    df.to_csv(output_csv, index=False)
