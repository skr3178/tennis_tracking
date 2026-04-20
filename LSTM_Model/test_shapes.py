"""Synthetic shape & flow tests for the WhereIsTheBall pipeline.

Run from the LSTM_Model/ directory:
    /media/skr/storage/ten_bad/.venv/bin/python test_shapes.py
"""

import torch

from lift_to_3d import lift_to_3d
from models.eot_network import EoTNetwork
from models.height_network import HeightNetwork, _DirectionalHeight
from models.lstm_blocks import FCHead, RecurrentLSTMStack, ResidualBiLSTMStack
from models.refinement_network import RefinementNetwork
from pipeline import WhereIsTheBall


def _ok(name, *parts):
    print(f"  [PASS] {name}: " + "  ".join(parts))


def test_residual_bilstm_stack():
    B, L = 4, 16
    for in_dim in (4, 5, 7):
        block = ResidualBiLSTMStack(in_dim=in_dim, hidden=64)
        x = torch.randn(B, L, in_dim)
        y = block(x)
        assert y.shape == (B, L, 128), f"in_dim={in_dim}: got {y.shape}"
        _ok(f"ResidualBiLSTMStack(in={in_dim})",
            f"{tuple(x.shape)} -> {tuple(y.shape)}")


def test_recurrent_lstm_stack():
    B = 4
    stack = RecurrentLSTMStack(in_dim=6, hidden=64)
    state = stack.init_state(B, device=torch.device("cpu"))
    x_t = torch.randn(B, 6)
    out, state = stack.step(x_t, state)
    assert out.shape == (B, 64)
    _ok("RecurrentLSTMStack.step", f"{tuple(x_t.shape)} -> {tuple(out.shape)}")


def test_fc_head():
    B, L = 4, 16
    head = FCHead(in_dim=128, out_dim=1, sigmoid=True)
    x = torch.randn(B, L, 128)
    y = head(x)
    assert y.shape == (B, L, 1)
    assert (y >= 0).all() and (y <= 1).all()
    _ok("FCHead(128->1, sigmoid)",
        f"{tuple(x.shape)} -> {tuple(y.shape)} (in [0,1])")

    head2 = FCHead(in_dim=64, out_dim=1, sigmoid=False)
    y2 = head2(torch.randn(B, L, 64))
    assert y2.shape == (B, L, 1)
    _ok("FCHead(64->1, no sig)", f"-> {tuple(y2.shape)}")

    head3 = FCHead(in_dim=128, out_dim=3, sigmoid=False)
    y3 = head3(torch.randn(B, L, 128))
    assert y3.shape == (B, L, 3)
    _ok("FCHead(128->3, no sig)", f"-> {tuple(y3.shape)}")


def test_lift_to_3d():
    B, L = 4, 16
    h = torch.rand(B, L, 1) * 2.0
    P = torch.randn(B, L, 4)
    P[..., 3] = P[..., 3].abs() + 0.5  # pv_y > 0 to avoid singular ray
    xyz = lift_to_3d(h, P)
    assert xyz.shape == (B, L, 3)
    assert torch.allclose(xyz[..., 1:2], h, atol=1e-5)  # y == h
    _ok("lift_to_3d",
        f"h{tuple(h.shape)}, P{tuple(P.shape)} -> xyz{tuple(xyz.shape)} (y==h)")


def test_eot_network():
    B, L = 4, 16
    net = EoTNetwork()
    dP = torch.randn(B, L, 4)
    eps = net(dP)
    assert eps.shape == (B, L, 1)
    assert (eps >= 0).all() and (eps <= 1).all()
    n_params = sum(p.numel() for p in net.parameters())
    _ok("EoTNetwork",
        f"dP{tuple(dP.shape)} -> eps{tuple(eps.shape)} ({n_params:,} params)")


def test_directional_height():
    B, L = 4, 16
    net = _DirectionalHeight()
    dP = torch.randn(B, L, 4)
    eps = torch.rand(B, L, 1)
    h_seq, delta_seq = net(dP, eps, reverse=False)
    assert h_seq.shape == (B, L, 1) and delta_seq.shape == (B, L, 1)
    assert torch.allclose(h_seq[:, 0], torch.zeros_like(h_seq[:, 0]))
    _ok("DirectionalHeight forward",
        f"-> h{tuple(h_seq.shape)}, h_0 == 0 verified")

    h_seq_b, _ = net(dP, eps, reverse=True)
    assert h_seq_b.shape == (B, L, 1)
    assert torch.allclose(h_seq_b[:, -1], torch.zeros_like(h_seq_b[:, -1]))
    _ok("DirectionalHeight backward",
        f"-> h{tuple(h_seq_b.shape)}, h_N == 0 verified")


def test_height_network():
    B, L = 4, 16
    net = HeightNetwork()
    dP = torch.randn(B, L, 4)
    P = torch.randn(B, L, 4)
    eps = torch.rand(B, L, 1)
    h_refined, h_combined = net(dP, eps, P)
    assert h_refined.shape == (B, L, 1)
    assert h_combined.shape == (B, L, 1)
    n_params = sum(p.numel() for p in net.parameters())
    _ok("HeightNetwork",
        f"-> h_refined{tuple(h_refined.shape)}, h_combined{tuple(h_combined.shape)} "
        f"({n_params:,} params)")


def test_refinement_network():
    B, L = 4, 16
    net = RefinementNetwork()
    r = torch.randn(B, L, 3)
    P = torch.randn(B, L, 4)
    delta = net(r, P)
    assert delta.shape == (B, L, 3)
    n_params = sum(p.numel() for p in net.parameters())
    _ok("RefinementNetwork",
        f"r{tuple(r.shape)}, P{tuple(P.shape)} -> delta{tuple(delta.shape)} "
        f"({n_params:,} params)")


def test_full_pipeline_shapes():
    for B, L in [(2, 8), (4, 64), (1, 122), (3, 256)]:
        torch.manual_seed(42)
        net = WhereIsTheBall()
        P = torch.randn(B, L, 4)
        P[..., 3] = P[..., 3].abs() + 0.5  # pv_y > 0
        out = net(P)
        assert out["xyz_final"].shape == (B, L, 3)
        assert out["eps"].shape == (B, L, 1)
        assert out["h_refined"].shape == (B, L, 1)
        assert out["h_combined"].shape == (B, L, 1)
        assert out["r"].shape == (B, L, 3)
        assert (out["eps"] >= 0).all() and (out["eps"] <= 1).all()
        _ok(f"Pipeline B={B}, L={L}",
            f"xyz{tuple(out['xyz_final'].shape)} "
            f"eps{tuple(out['eps'].shape)} "
            f"h_ref{tuple(out['h_refined'].shape)}")


def test_mask_equivalence():
    """Same sequence run alone vs padded with a longer one must produce
    identical predictions on the valid frames when `lengths` is supplied."""
    torch.manual_seed(123)
    net = WhereIsTheBall().eval()
    ls, ll = 17, 41
    Ps = torch.randn(ls, 4); Ps[:, 3] = Ps[:, 3].abs() + 0.5
    Pl = torch.randn(ll, 4); Pl[:, 3] = Pl[:, 3].abs() + 0.5

    P_alone = Ps.unsqueeze(0)
    with torch.no_grad():
        out_alone = net(P_alone, lengths=torch.tensor([ls]))

    P_pad = torch.zeros(2, ll, 4)
    P_pad[0, :ls] = Ps; P_pad[1] = Pl
    with torch.no_grad():
        out_pad = net(P_pad, lengths=torch.tensor([ls, ll]))

    for k in ['eps', 'h_refined', 'xyz_final', 'r', 'h_combined']:
        d = (out_alone[k][0] - out_pad[k][0, :ls]).abs().max().item()
        assert d < 1e-4, f"{k}: alone vs padded diverge by {d}"
    _ok("mask-equivalence (alone vs padded)",
        f"max diff < 1e-4 on {len(['eps','h_refined','xyz_final','r','h_combined'])} outputs")


def test_full_pipeline_backward():
    """Smoke-test gradient flow end-to-end."""
    B, L = 2, 32
    torch.manual_seed(0)
    net = WhereIsTheBall()
    P = torch.randn(B, L, 4)
    P[..., 3] = P[..., 3].abs() + 0.5
    out = net(P)
    loss = out["xyz_final"].pow(2).mean() + out["eps"].mean()
    loss.backward()
    n_with_grad = sum(1 for p in net.parameters() if p.grad is not None
                      and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in net.parameters())
    assert n_with_grad == n_total, f"only {n_with_grad}/{n_total} params got grads"
    _ok("end-to-end backward",
        f"loss={loss.item():.4f}, grads on {n_with_grad}/{n_total} params")


def main():
    torch.manual_seed(0)
    print("\n=== Building blocks ===")
    test_residual_bilstm_stack()
    test_recurrent_lstm_stack()
    test_fc_head()
    test_lift_to_3d()

    print("\n=== Sub-networks ===")
    test_eot_network()
    test_directional_height()
    test_height_network()
    test_refinement_network()

    print("\n=== Full pipeline (varied B, L) ===")
    test_full_pipeline_shapes()

    print("\n=== Variable-length / masking ===")
    test_mask_equivalence()

    print("\n=== Backward pass ===")
    test_full_pipeline_backward()

    print("\nAll shape tests passed.\n")


if __name__ == "__main__":
    main()
