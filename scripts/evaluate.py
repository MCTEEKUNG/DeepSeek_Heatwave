"""
เมตริกกลางสำหรับประเมินโมเดล (เชิงความน่าจะเป็น) — ใช้ร่วมกันทุกโมเดลเพื่อเทียบกันยุติธรรม

นิยามความสำเร็จของงาน: ต้อง "ชนะ baseline climatology" (BSS > 0)
- คลื่นความร้อนเป็นเหตุการณ์หายาก -> accuracy ไม่มีความหมาย, ต้องดู Brier/BSS/AUC/reliability
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score = ค่าเฉลี่ย (p - y)^2 ; ต่ำ = ดี."""
    return float(brier_score_loss(y_true, y_prob))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """BSS เทียบ baseline climatology (พยากรณ์ด้วยอัตราฐานคงที่).

    BSS = 1 - BS_model / BS_climatology
      > 0  -> ชนะ climatology (ดี)
      = 0  -> เท่า climatology
      < 0  -> แย่กว่า climatology
    """
    y_true = np.asarray(y_true, dtype=float)
    base_rate = y_true.mean()
    bs_model = np.mean((y_prob - y_true) ** 2)
    bs_clim = np.mean((base_rate - y_true) ** 2)
    if bs_clim == 0:  # ไม่มีความแปรผัน (ทุกค่าเหมือนกัน) -> นิยามไม่ได้
        return float("nan")
    return float(1.0 - bs_model / bs_clim)


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC ; ถ้ามีคลาสเดียว (ไม่มีเหตุการณ์เลย) -> nan."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """ข้อมูลสำหรับ reliability diagram: (ความน่าจะเป็นเฉลี่ยที่ทำนาย, ความถี่จริงที่สังเกต, จำนวน) ต่อ bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b] > 0:
            mean_pred[b] = y_prob[m].mean()
            obs_freq[b] = y_true[m].mean()
    return mean_pred, obs_freq, counts


def evaluate_probabilistic(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """รวมเมตริกหลักไว้ใน dict เดียว (ใช้บันทึกผลต่อ run)."""
    y_true = np.asarray(y_true)
    return {
        "n": int(y_true.size),
        "base_rate": float(np.mean(y_true)),
        "brier": brier(y_true, y_prob),
        "bss": brier_skill_score(y_true, y_prob),
        "auc": auc(y_true, y_prob),
    }


def _selftest() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    base = 0.1
    y = (rng.random(n) < base).astype(int)

    # 1) พยากรณ์สมบูรณ์แบบ: p = y -> Brier 0, BSS 1, AUC 1
    perfect = evaluate_probabilistic(y, y.astype(float))
    assert abs(perfect["brier"]) < 1e-9, perfect
    assert abs(perfect["bss"] - 1.0) < 1e-9, perfect
    assert abs(perfect["auc"] - 1.0) < 1e-9, perfect
    print("พยากรณ์สมบูรณ์แบบ:", {k: round(v, 4) for k, v in perfect.items()})

    # 2) พยากรณ์ climatology (อัตราฐานคงที่) -> BSS ~ 0
    clim = evaluate_probabilistic(y, np.full(n, y.mean()))
    assert abs(clim["bss"]) < 1e-6, clim
    print("พยากรณ์ climatology  :", {k: round(v, 4) for k, v in clim.items()})

    # 3) พยากรณ์มั่ว -> AUC ~ 0.5, BSS <= ~0
    rand = evaluate_probabilistic(y, rng.random(n))
    print("พยากรณ์มั่ว          :", {k: round(v, 4) for k, v in rand.items()})
    assert 0.4 < rand["auc"] < 0.6, rand

    # 4) reliability: ผลรวม counts = n
    _, _, counts = reliability_curve(y, rng.random(n), n_bins=10)
    assert counts.sum() == n
    print("reliability bins รวม :", int(counts.sum()), "= n ✅")
    print("[OK] เมตริกทั้งหมดถูกต้องตามค่าที่ทราบ")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
