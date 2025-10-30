#!/usr/bin/env python3
"""
Quick mock forward for WanModel (text->video) to inspect tensor shapes and timing.
This instantiates a tiny model (small dims / few layers) to avoid heavy GPU memory.
Run with: python3 scripts/mock_forward_t2v.py
"""
import time
import torch

from wan.modules.model import WanModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Small model config to keep runtime tiny and inspect shapes
    model = WanModel(
        model_type='t2v',
        patch_size=(1, 2, 2),
        in_dim=1,
        dim=64,
        ffn_dim=256,
        freq_dim=64,
        text_dim=128,
        out_dim=1,
        num_heads=8,
        num_layers=2,
    ).to(device)
    model.eval()

    # Input video: [C_in, F, H, W]
    C_in = model.in_dim
    F = 5
    H = 8
    W = 8
    x = [torch.randn(C_in, F, H, W, device=device)]

    # seq_len is number of patches (F_patches * H_patches * W_patches)
    p_t, p_h, p_w = model.patch_size
    Fp = F // p_t
    Hp = H // p_h
    Wp = W // p_w
    seq_len = Fp * Hp * Wp

    # timestep (batch size 1)
    t = torch.tensor([10], device=device)

    # context: list of text embedding tensors of shape [L, text_dim]
    L = 16
    context = [torch.randn(L, model.text_dim, device=device)]

    print("Model params:", sum(p.numel() for p in model.parameters()))
    print("Input shapes:")
    print("  x[0]", x[0].shape)
    print("  seq_len", seq_len)
    print("  t", t.shape, t)
    print("  context[0]", context[0].shape)

    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(x, t, context, seq_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    print(f"Forward time: {t1 - t0:.6f}s")
    print("Output:")
    print("  len(out)", len(out))
    print("  out[0] shape", out[0].shape)

    # estimate memory of the input / output tensors (bytes)
    def tensor_bytes(t):
        return t.numel() * t.element_size()

    in_bytes = tensor_bytes(x[0])
    out_bytes = tensor_bytes(out[0])
    print(f"Estimated input memory: {in_bytes} bytes ({in_bytes/1024:.1f} KiB)")
    print(f"Estimated output memory: {out_bytes} bytes ({out_bytes/1024:.1f} KiB)")


if __name__ == '__main__':
    main()
