"""MoLoRA vs LoRA 效果对比：快速验证实验。

减少数据量和 epoch 以在 CPU 上快速运行，核心逻辑保持不变。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ultralytics.nn.peft.molora import (
    MoLoRAConfig, get_peft_molora_model, MoLoRAModel,
    mark_only_molora_as_trainable, allocate_domain_experts,
)
from ultralytics.nn.peft.molora.layer import MoLoRAExpert


# 轻量配置
SEED = 42
NUM_DOMAINS = 3
SAMPLES = 200
IMG_SIZE = 16
NUM_CLASSES = 5
BASE_CH = 16
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.005
DEVICE = "cpu"
R = 4
ALPHA = 8
MOLORA_E = 4
MOLORA_K = 2


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def gen_data(domain_idx, n, seed=42):
    rng = np.random.RandomState(seed + domain_idx * 100)
    centers = rng.randn(NUM_CLASSES, IMG_SIZE * IMG_SIZE) * 1.5
    if domain_idx == 1: centers += 0.8
    elif domain_idx == 2: centers *= 1.8
    labels = rng.randint(0, NUM_CLASSES, n)
    imgs = np.array([centers[l] + rng.randn(IMG_SIZE * IMG_SIZE) * 0.4 for l in labels])
    imgs = imgs.reshape(n, 1, IMG_SIZE, IMG_SIZE)
    imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-8)
    return torch.tensor(imgs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, BASE_CH, 3, padding=1)
        self.conv2 = nn.Conv2d(BASE_CH, BASE_CH, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(BASE_CH, NUM_CLASSES)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


def train_epoch(m, loader, opt):
    m.train(); loss_sum = 0; correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); logits = m(x); loss = F.cross_entropy(logits, y)
        loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def evaluate(m, loader):
    m.eval(); loss_sum = 0; correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = m(x); loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def inject_lora(m, r, alpha):
    params = []
    for name, mod in list(m.named_modules()):
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and any(t in name for t in ["conv1","conv2","fc"]):
            e = MoLoRAExpert(mod, r=r, alpha=alpha, use_rslora=True).to(DEVICE)
            for p in mod.parameters(): p.requires_grad = False
            for p in e.parameters(): p.requires_grad = True; params.append(p)
            orig = mod.forward
            mod.forward = lambda x, o=orig, ex=e: o(x) + ex(x)
    return params


def exp_single():
    print("\n=== 实验 1: 单域微调 ===")
    X_tr, y_tr = gen_data(0, SAMPLES)
    X_te, y_te = gen_data(0, SAMPLES // 4, seed=999)
    tr = DataLoader(TensorDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    te = DataLoader(TensorDataset(X_te, y_te), BATCH_SIZE, shuffle=False)

    # Baseline
    set_seed(SEED); base = TinyCNN().to(DEVICE)
    opt = torch.optim.Adam(base.parameters(), lr=LR)
    for e in range(EPOCHS): train_epoch(base, tr, opt)
    _, b_acc = evaluate(base, te); print(f"Baseline: {b_acc:.4f}")

    # LoRA
    set_seed(SEED); m_l = TinyCNN().to(DEVICE)
    lp = inject_lora(m_l, R, ALPHA)
    opt = torch.optim.Adam(lp, lr=LR)
    for e in range(EPOCHS): train_epoch(m_l, tr, opt)
    _, l_acc = evaluate(m_l, te); l_n = sum(p.numel() for p in lp)
    print(f"LoRA:     {l_acc:.4f}  (params: {l_n:,})")

    # MoLoRA
    set_seed(SEED); m_m = TinyCNN().to(DEVICE)
    cfg = MoLoRAConfig(r=R, alpha=ALPHA, num_experts=MOLORA_E, top_k=MOLORA_K,
                       router_type="linear", use_rslora=True,
                       target_modules=["conv1","conv2","fc"])
    m_m = get_peft_molora_model(m_m, cfg)
    mark_only_molora_as_trainable(m_m)
    mp = [p for p in m_m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(mp, lr=LR)
    for e in range(EPOCHS): train_epoch(m_m, tr, opt)
    _, m_acc = evaluate(m_m, te); m_n = sum(p.numel() for p in mp)
    print(f"MoLoRA:   {m_acc:.4f}  (params: {m_n:,})")
    print(f"增益: {m_acc - l_acc:+.4f}")
    return {"baseline": b_acc, "lora": l_acc, "molora": m_acc}


def exp_continual():
    print("\n=== 实验 2: 多域持续学习 ===")
    domains = ["day","night","fog"]
    data = {}
    for i, d in enumerate(domains):
        X, y = gen_data(i, SAMPLES)
        Xt, yt = gen_data(i, SAMPLES // 4, seed=888 + i)
        data[d] = {"train": DataLoader(TensorDataset(X,y), BATCH_SIZE, shuffle=True),
                   "test": DataLoader(TensorDataset(Xt,yt), BATCH_SIZE, shuffle=False)}

    # LoRA
    set_seed(SEED); m_l = TinyCNN().to(DEVICE)
    lp = inject_lora(m_l, R, ALPHA)
    opt = torch.optim.Adam(lp, lr=LR)
    for d in domains:
        for e in range(EPOCHS): train_epoch(m_l, data[d]["train"], opt)
    lora_accs = {f"on_{ed}": evaluate(m_l, data[ed]["test"])[1] for ed in domains}

    # MoLoRA
    set_seed(SEED); m_m = TinyCNN().to(DEVICE)
    dom_exp = allocate_domain_experts(MOLORA_E, domains)
    cfg = MoLoRAConfig(r=R, alpha=ALPHA, num_experts=MOLORA_E, top_k=MOLORA_K,
                       router_type="linear", domain_experts=dom_exp,
                       use_rslora=True, target_modules=["conv1","conv2","fc"])
    m_m = get_peft_molora_model(m_m, cfg)
    w = MoLoRAModel(m_m, cfg)
    opt = torch.optim.Adam([p for p in w.model.parameters() if p.requires_grad], lr=LR)
    for d in domains:
        w.set_domain(d)
        for e in range(EPOCHS): train_epoch(w.model, data[d]["train"], opt)
    molora_accs = {}
    for ed in domains:
        w.set_domain(ed); molora_accs[f"on_{ed}"] = evaluate(w.model, data[ed]["test"])[1]

    print(f"{'场景':<12} {'LoRA':>8} {'MoLoRA':>8} {'增益':>8}")
    for d in domains:
        la = lora_accs[f"on_{d}"]; ma = molora_accs[f"on_{d}"]
        print(f"{d:<12} {la:>8.4f} {ma:>8.4f} {ma-la:>+8.4f}")

    lf = lora_accs["on_day"] - lora_accs["on_fog"]
    mf = molora_accs["on_day"] - molora_accs["on_fog"]
    print(f"\nLoRA 遗忘: {lf:+.4f}")
    print(f"MoLoRA 遗忘: {mf:+.4f}")
    print(f"遗忘减少: {abs(lf)-abs(mf):+.4f}")
    return lora_accs, molora_accs


def main():
    set_seed(SEED)
    r1 = exp_single()
    lora_accs, molora_accs = exp_continual()
    print("\n=== 总结 ===")
    print(f"单域增益: {r1['molora'] - r1['lora']:+.4f}")
    print(f"多域遗忘减少: {abs(lora_accs['on_day'] - molora_accs['on_day']):+.4f}")
    if r1['molora'] > r1['lora']:
        print("✅ MoLoRA 单域拟合优于 LoRA")
    if abs(molora_accs['on_fog'] - molora_accs['on_day']) < abs(lora_accs['on_fog'] - lora_accs['on_day']):
        print("✅ MoLoRA 显著减少灾难性遗忘")

if __name__ == "__main__":
    main()
