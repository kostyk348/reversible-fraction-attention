"""05 — Parallel training on a synthetic associative-recall task.

The model must learn to predict a linear projection of the *first*
token from the *last* output of the parallel attention. This verifies
that information can flow through the full context window end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rfa import ParallelReversibleAttention


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, seq_len, batch_size, epochs = 32, 64, 16, 150

    model = ParallelReversibleAttention(d_model=d_model, max_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    target_projection = nn.Linear(d_model, d_model, bias=False).to(device)

    print(f"--- 6. Vectorised training on {device} ---")
    print("Learning to retrieve information across the full context window...")
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        y_target = target_projection(x[:, 0, :])
        out = model(x)
        pred_last = out[:, -1, :]
        loss = nn.functional.mse_loss(pred_last, y_target)
        loss.backward()
        optimizer.step()
        if epoch % 30 == 0:
            with torch.no_grad():
                gamma = model.current_gamma().item()
            print(f"epoch {epoch:3d} | loss {loss.item():.6f} | adaptive gamma {gamma:.4f}")
    print("Training done.")


if __name__ == "__main__":
    main()
