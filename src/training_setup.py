import torch
from torch.optim.lr_scheduler import LambdaLR


def make_optimizer_and_scheduler(model, learning_rate, warmup_steps, weight_decay, warmup_init_lr=1e-7):
    # Adam with decoupled weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def lr_lambda(step):  # multiplier on learning_rate
        if step < warmup_steps:  # linear ramp from warmup_init_lr up to learning_rate
            return (warmup_init_lr + step * (learning_rate - warmup_init_lr) / warmup_steps) / learning_rate
        return 1.0  # constant after warmup

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler