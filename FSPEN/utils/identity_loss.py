import torch
import torch.nn as nn
from speechbrain.inference.speaker import SpeakerRecognition


class SpeakerIdentityLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Load pre-trained ECAPA model
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        self.verification.eval()  # Freeze verification model
        for param in self.verification.parameters():
            param.requires_grad = False

    def forward(self, enhanced_wav, clean_wav):
        """Calculates 1 - Cosine Similarity between embeddings"""
        # wav shape expected: (Batch, time_samples)
        emb_enh = self.verification.encode_batch(enhanced_wav).squeeze(1)
        emb_cln = self.verification.encode_batch(clean_wav).squeeze(1)

        # Normalize
        emb_enh = torch.nn.functional.normalize(emb_enh, dim=-1)
        emb_cln = torch.nn.functional.normalize(emb_cln, dim=-1)

        # Loss: 1 - cosine similarity (minimize this to maximize similarity)
        cos_sim = (emb_enh * emb_cln).sum(dim=-1)
        return 1 - cos_sim.mean()