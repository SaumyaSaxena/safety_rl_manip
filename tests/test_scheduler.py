from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda", 0)
    layer = nn.Linear(
        in_features=128,
        out_features=256,
    ).to(device=device)

    optimizer = optim.AdamW(layer.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    all_lr = []
    for i in range(50):
        scheduler.step()
        all_lr.append(scheduler.get_last_lr())

    fig, axes = plt.subplots(1, figsize=(16, 16))

    # plot position
    axes.plot(np.arange(len(all_lr)), all_lr, label='EE x')
    plt.savefig('test_scheduler.png')

    plt.close()