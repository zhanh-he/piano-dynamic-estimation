# -----------------------------------------------------------------------------
# NOTE: Upstream Code Acknowledgement (adapted)
# Source project : Guzheng ICASSP 2023
# Source repo    : https://github.com/LiDCC/GuzhengTech99
# Code path      : https://github.com/LiDCC/GuzhengTech99/blob/main/function/model.py
# Inspiration    : Multi-scale attention CNN design
# Modification   : Adapted/extended for this project (temporal strides s1/s2, heads, etc.)
# Thanks to the Guzheng authors for releasing the original implementation.
# License        : See the upstream repository for license terms.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import get_feature_extractor_and_bins
from collections.abc import Sequence
"""
References: Guzheng ICASSP2023 model: https://github.com/LiDCC/GuzhengTech99/blob/main/function/model.py
"""
class block(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inp)       
        self.conv1 = nn.Conv2d(inp, out, (3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(out)       
        self.conv2 = nn.Conv2d(out, out, (3,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(out)       

        self.sk = nn.Conv2d(inp, out, (1,1), padding=(0,0))

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        out += self.sk(x)
        return out

class self_att(nn.Module):
    def __init__(self, embeded_dim, num_heads, out):
        super().__init__()
        self.att = nn.MultiheadAttention(embeded_dim, num_heads, batch_first=True)

    def forward(self, x):
        # Squeeze last dim only; keep batch
        x1 = x.squeeze(-1).transpose(-1, -2)
        res_branch, attn_wei = self.att(x1, x1, x1)
        res = res_branch.transpose(-1,-2).unsqueeze(-1)
        res = torch.add(res, x)
        return res

class SYMultiScaleAttnFlex(nn.Module):
    """Multi-scale attn CNN with two temporal strides (s1, s2)."""
    def __init__(self, FRE, latent_dim, temporal_strides=None):
        super().__init__()
        inp = FRE
        size = 1
        fs = (3, 1)
        ps = (1, 0)

        if temporal_strides is None:
            s1, s2 = 3, 3
        else:
            # Accept list/tuple (non-string Sequence)
            if isinstance(temporal_strides, Sequence) and not isinstance(temporal_strides, (str, bytes)):
                s1 = int(temporal_strides[0]) if len(temporal_strides) >= 1 else 1
                s2 = int(temporal_strides[1]) if len(temporal_strides) >= 2 else 1
            else:
                s1 = s2 = int(temporal_strides)
            s1 = max(1, s1); s2 = max(1, s2)

        self.s1, self.s2 = s1, s2

        self.bn0   = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, FRE, (1, size), padding=(0, 0))

        self.maxpool012 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.conv02     = nn.Conv2d(inp, FRE, (1, size), padding=(0, 0))

        self.block11 = block(FRE, FRE * 2)
        self.block12 = block(FRE, FRE * 2)

        self.maxpool112 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.dropout12  = nn.Dropout(p=0.2)
        self.maxpool123 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.dropout123 = nn.Dropout(p=0.2)
        self.us121      = nn.ConvTranspose2d(FRE*2, FRE*2, kernel_size=(s1, 1), stride=(s1, 1))

        self.conv21 = nn.Conv2d(FRE*2, FRE*2, (1, 2))
        self.conv22 = nn.Conv2d(FRE*2, FRE*2, (1, 2))
        self.conv23 = nn.Conv2d(FRE*2, FRE*2, (1, 1))

        self.block21 = block(FRE * 2, FRE * 3)
        self.block22 = block(FRE * 2, FRE * 3)
        self.block23 = block(FRE * 2, FRE * 3)

        self.self_att23 = self_att(FRE * 3, 1, 1)
        self.bn23       = nn.BatchNorm2d(FRE * 3)

        self.maxpool212 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.maxpool223 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.dropout22  = nn.Dropout(p=0.2)
        self.dropout23  = nn.Dropout(p=0.2)
        self.us221      = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(s2, 1), stride=(s2, 1))
        self.us232      = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(s2, 1), stride=(s2, 1))

        self.conv31 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))
        self.conv32 = nn.Conv2d(FRE * 3, FRE * 3, (1, 3))
        self.conv33 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))

        self.block31 = block(FRE * 3, FRE * 3)
        self.block32 = block(FRE * 3, FRE * 3)
        self.block33 = block(FRE * 3, FRE * 3)

        self.bn31  = nn.BatchNorm2d(FRE * 3); self.relu31 = nn.ReLU(inplace=True)
        self.bn32  = nn.BatchNorm2d(FRE * 3); self.relu32 = nn.ReLU(inplace=True)
        self.bn33  = nn.BatchNorm2d(FRE * 3); self.relu33 = nn.ReLU(inplace=True)

        self.maxpool312 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.dropout312 = nn.Dropout(p=0.2)
        self.us321      = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(s1, 1), stride=(s1, 1))
        self.us332      = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(s2, 1), stride=(s2, 1))

        self.self_att = self_att(FRE * 3, 1, 1)
        self.bn4      = nn.BatchNorm2d(FRE * 3)

        self.conv41 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))
        self.conv42 = nn.Conv2d(FRE * 3, FRE * 3, (1, 3))

        self.block41 = block(FRE * 3, FRE * 2)
        self.block42 = block(FRE * 3, FRE * 2)

        self.us421 = nn.ConvTranspose2d(FRE * 2, FRE * 2, kernel_size=(s1, 1), stride=(s1, 1))

        self.conv51 = nn.Conv2d(FRE * 2, FRE * 2, (1, 2))
        self.block51 = block(FRE * 2, FRE)

        self.bn51 = nn.BatchNorm2d(FRE); self.relu51 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(FRE, latent_dim, fs, padding=ps)

    def forward(self, x, Xavg=None, Xstd=None):
        x0  = self.bn0(x)
        x02 = self.maxpool012(x0)
        x02 = self.block12(self.conv02(x02))

        x1  = self.conv1(x0)
        x11 = self.block11(x1)

        x11_n = torch.cat([x11, self.us121(x02, output_size=x11.shape)], dim=-1)

        x12   = self.dropout12(self.maxpool112(x11))
        x12_n = torch.cat([x02, x12], dim=-1)
        x13   = self.dropout123(self.maxpool123(x02))

        x21 = self.block21(self.conv21(x11_n))
        x22 = self.block22(self.conv22(x12_n))
        x23 = self.block23(self.conv23(x13))

        x21_n = torch.cat([x21, self.us221(x22, output_size=x21.shape)], dim=-1)
        x22_n = torch.cat([x22,
                           self.dropout22(self.maxpool212(x21)),
                           self.us232(x23, output_size=x22.shape)], dim=-1)
        x23   = self.bn23(self.self_att23(x23))
        x23_n = torch.cat([x23, self.dropout23(self.maxpool223(x22))], dim=-1)

        x31 = self.relu31(self.bn31(self.block31(self.conv31(x21_n))))
        x32 = self.relu32(self.bn32(self.block32(self.conv32(x22_n))))
        x33 = self.relu33(self.bn33(self.block33(self.conv33(x23_n))))
        x33 = self.bn4(self.self_att(x33))

        x31_n = torch.cat([x31, self.us321(x32, output_size=x31.shape)], dim=-1)
        x32_n = torch.cat([x32,
                           self.us332(x33, output_size=x32.shape),
                           self.dropout312(self.maxpool312(x31))], dim=-1)

        x41 = self.block41(self.conv41(x31_n))
        x42 = self.block42(self.conv42(x32_n))

        x51_n = torch.cat([x41, self.us421(x42, output_size=x41.shape)], dim=-1)
        x51   = self.relu51(self.bn51(self.block51(self.conv51(x51_n))))

        pred = self.conv4(x51)
        return pred


class SingleCNN(nn.Module):
    """
    Single-target CNN head over audio features.
    """
    def __init__(self, cfg):
        super().__init__()
        # Feature params
        sample_rate         = cfg.feature.sample_rate
        fft_size            = cfg.feature.fft_size
        frames_per_second   = cfg.feature.frames_per_second
        audio_feature       = cfg.feature.audio_feature

        # Target: multi-class 'dynamic' or one binary task
        self.target = cfg.exp.targets[0] if hasattr(cfg, 'exp') and hasattr(cfg.exp, 'targets') else 'dynamic'
        if self.target == 'dynamic':
            classes_num = cfg.feature.dynamic_classes + 1  # include blank
        elif self.target in ('beat', 'downbeat', 'change_point'):
            classes_num = 1  # binary logit head
        else:
            raise ValueError(f"Unsupported single target: {self.target}")

        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(
            audio_feature, sample_rate, fft_size, frames_per_second)

        self.adapt_conv = None  # lazy 1x1 conv for channel mismatch

        self.bn0 = nn.BatchNorm2d(self.FRE, momentum=0.01)

        # Encoder with fixed [3,3] temporal strides
        self.encoder = SYMultiScaleAttnFlex(self.FRE, classes_num, temporal_strides=[3, 3])

    def forward(self, input, target_len=None):
        x = self.feature_extractor(input)
        # Accept [B,F,T] or [F,T]; replicate batch if collapsed
        B_in = input.shape[0] if hasattr(input, 'shape') and input.dim() >= 1 else 1
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(B_in, 1, 1)
        if x.dim() != 3:
            raise RuntimeError(f"Unexpected feature shape {tuple(x.shape)}; expected 3D [B, F, T] or [B, T, F]")

        # If channels are last, transpose to [B,F,T]
        if x.shape[-1] == self.FRE and x.shape[1] != self.FRE:
            x = x.transpose(1, 2)

        F_actual = x.shape[1]
        if F_actual != self.FRE:
            # Case A: expand single-channel to FRE
            if F_actual == 1 and self.FRE > 1:
                x = x.repeat(1, self.FRE, 1)
            # Case B: collapse many channels to single
            elif self.FRE == 1 and F_actual > 1:
                x = x.mean(dim=1, keepdim=True)
            else:
                # Learn 1x1 Conv to map channels
                if (self.adapt_conv is None or
                    getattr(self.adapt_conv, 'in_channels', None) != F_actual or
                    getattr(self.adapt_conv, 'out_channels', None) != self.FRE):
                    self.adapt_conv = nn.Conv1d(F_actual, self.FRE, kernel_size=1, bias=False).to(x.device)
                x = self.adapt_conv(x)

        # To 4D for 2D-CNN blocks: [B,F,T,1]
        x = x.unsqueeze(3)
        x = self.bn0(x)
        x = self.encoder(x, None, None)
        x = x.squeeze(-1).transpose(1, 2)
        if target_len is not None:
            x = x[:, :target_len, :]
        return {f'{self.target}_output': x}

# ---------------- MMoE ----------------

class ExpertConv1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1), nn.ReLU(),
            nn.Conv1d(c, c, 3, padding=1))
    def forward(self, x):  # x: [B,T,C]
        return self.net(x.transpose(1, 2)).transpose(1, 2)

class MMoEBlock(nn.Module):
    def __init__(self, c, n_experts=4, n_tasks=4):
        super().__init__()
        self.experts = nn.ModuleList([ExpertConv1D(c) for _ in range(n_experts)])
        self.gates   = nn.ModuleList([nn.Linear(c, n_experts) for _ in range(n_tasks)])
    def forward(self, feat):  # [B,T,C]
        Es = torch.stack([E(feat) for E in self.experts], dim=2)  # [B,T,E,C]
        outs = []
        gates_per_task = []
        for g in self.gates:
            w = g(feat).softmax(dim=-1)                           # [B,T,E]
            outs.append((Es * w.unsqueeze(-1)).sum(dim=2))        # [B,T,C]
            gates_per_task.append(w)                              # [B,T,E]
        return outs, gates_per_task  # len= n_tasks


class MultiTaskCNN(nn.Module):
    """Proposed Multitask Multiscale network, an CNN encoder + MMoE with heads for dynamic/beat/downbeat/change_point."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        sr    = cfg.feature.sample_rate
        nfft  = cfg.feature.fft_size
        fps   = cfg.feature.frames_per_second
        afe   = cfg.feature.audio_feature

        # Feature extractor + encoder
        self.extractor, self.FRE = get_feature_extractor_and_bins(afe, sr, nfft, fps)
        cnn_cfg = getattr(cfg, 'cnn', None)
        # Prefer cnn.latent_dim; fallback to legacy names
        latent_dim = int(getattr(cnn_cfg, 'latent_dim', getattr(cnn_cfg, 'out_channels', getattr(cnn_cfg, 'feat_dim', getattr(cnn_cfg, 'channels', 96)))))
        self.bn0 = nn.BatchNorm2d(self.FRE, momentum=0.01)

        # Encoder (flex): prefer cnn.temporal_scale; fallback to cnn.temporal_strides
        temporal_strides = getattr(cnn_cfg, 'temporal_scale', getattr(cnn_cfg, 'temporal_strides', [3, 3]))
        self.encoder = SYMultiScaleAttnFlex(self.FRE, latent_dim, temporal_strides=temporal_strides)

        # Shared -> task projections
        self.drop = nn.Dropout(float(getattr(cnn_cfg, 'fc_dropout', 0.2)))
        self.proj_dyn  = nn.Linear(latent_dim, latent_dim)
        self.proj_beat = nn.Linear(latent_dim, latent_dim)
        self.proj_db   = nn.Linear(latent_dim, latent_dim)
        self.proj_cp   = nn.Linear(latent_dim, latent_dim)

        # MMoE
        self.use_mmoe = bool(getattr(cnn_cfg, 'use_mmoe', False))
        if self.use_mmoe:
            n_exp = int(getattr(cnn_cfg, 'n_experts', 4))
            self.mmoe = MMoEBlock(latent_dim, n_experts=n_exp, n_tasks=4)
        # Export expert heatmap during inference
        self.export_mmoe_heatmap = bool(getattr(cnn_cfg, 'export_mmoe_heatmap', False))
        self._last_mmoe_gates = None  # set in forward when export flag is on

        # Task heads
        K = int(cfg.feature.dynamic_classes) + 1
        self.head_dyn_out  = nn.Linear(latent_dim, K)
        self.head_beat_out = nn.Linear(latent_dim, 1)
        self.head_db_out   = nn.Linear(latent_dim, 1)
        self.head_cp_out   = nn.Linear(latent_dim, 1)

        # init
        for m in [self.proj_dyn, self.proj_beat, self.proj_db, self.proj_cp,
                  self.head_dyn_out, self.head_beat_out, self.head_db_out,
                  self.head_cp_out]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0.0)

    def forward(self, input, target_len=None):
        # Feature & encoder
        x = self.extractor(input)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == self.FRE and x.shape[1] != self.FRE:
            x = x.transpose(1, 2)
        x = x.unsqueeze(3)
        x = self.bn0(x)
        feat = self.encoder(x, None, None).squeeze(-1).transpose(1, 2)
        feat = self.drop(feat)

        # Task-specific features
        h_dyn  = F.gelu(self.proj_dyn(feat))
        h_beat = F.gelu(self.proj_beat(feat))
        h_db   = F.gelu(self.proj_db(feat))
        h_cp   = F.gelu(self.proj_cp(feat))

        if self.use_mmoe:
            mmoe_out = self.mmoe(feat)
            if isinstance(mmoe_out, tuple):
                (h_dyn, h_beat, h_db, h_cp), gates_per_task = mmoe_out
                if self.export_mmoe_heatmap:
                    # Save gates for retrieval: list of 4 tensors [B,T,E]
                    self._last_mmoe_gates = gates_per_task
            else:
                h_dyn, h_beat, h_db, h_cp = mmoe_out

        # Logits
        dyn_logits  = self.head_dyn_out(h_dyn)
        beat_logits = self.head_beat_out(h_beat)
        db_logits   = self.head_db_out(h_db)
        cp_logits   = self.head_cp_out(h_cp)

        # Align to target_len
        if target_len is not None and dyn_logits.size(1) != target_len:
            def pad_or_cut(t, C):
                T = t.size(1)
                if T > target_len:
                    return t[:, :target_len, :C]
                pad = t.new_zeros(t.size(0), target_len - T, C)
                return torch.cat([t, pad], dim=1)
            dyn_logits  = pad_or_cut(dyn_logits,  dyn_logits.size(-1))
            beat_logits = pad_or_cut(beat_logits, 1)
            db_logits   = pad_or_cut(db_logits,   1)
            cp_logits   = pad_or_cut(cp_logits,   1)

        return {
            'dynamic_output':      dyn_logits,
            'beat_output':         beat_logits,
            'downbeat_output':     db_logits,
            'change_point_output': cp_logits,
        }
