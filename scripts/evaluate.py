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


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline_prob: float | np.ndarray | None = None,
) -> float:
    """BSS เทียบ baseline ที่ระบุ.

    BSS = 1 - BS_model / BS_baseline
      > 0  -> ชนะ baseline (ดี)
      = 0  -> เท่า baseline
      < 0  -> แย่กว่า baseline

    baseline_prob:
      - scalar : ความน่าจะเป็นคงที่ เช่น base rate จาก "ชุด train" (climatology ที่ยุติธรรม)
      - array  : baseline รายตัวอย่าง เช่น seasonal climatology / persistence
      - None   : ใช้ base rate ของ y_true เอง — baseline "เห็นเฉลย" ช่วง test
                 ทำให้ BSS ถูกกดต่ำกว่าจริง เหมาะกับ sanity check เท่านั้น
                 (รายงานจริงต้องส่ง baseline จากชุด train เสมอ)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if baseline_prob is None:
        baseline = np.full_like(y_true, y_true.mean())
    else:
        baseline = np.broadcast_to(np.asarray(baseline_prob, dtype=float), y_true.shape)
    bs_model = np.mean((y_prob - y_true) ** 2)
    bs_ref = np.mean((baseline - y_true) ** 2)
    if bs_ref == 0:  # baseline สมบูรณ์แบบ/ไม่มีความแปรผัน -> นิยามไม่ได้
        return float("nan")
    return float(1.0 - bs_model / bs_ref)


def seasonal_climatology(
    doy_train: np.ndarray, y_train: np.ndarray, window: int = 15
) -> np.ndarray:
    """ความน่าจะเป็นของเหตุการณ์ต่อ "วันในปี" จากชุด train (baseline ที่แกร่งกว่า base rate คงที่).

    สำหรับวันในปี d: P(event) = ความถี่เฉลี่ยของ y_train ในหน้าต่าง ±window วัน (วนรอบปฏิทิน)
    คืน array ยาว 367 (index ตรงกับ doy 1..366 ; index 0 ไม่ใช้)
    วันที่ไม่มีตัวอย่างในหน้าต่าง -> ถอยไปใช้ base rate รวมของ train
    """
    doy_train = np.asarray(doy_train, dtype=int)
    y_train = np.asarray(y_train, dtype=float)
    base = float(y_train.mean()) if y_train.size else float("nan")
    probs = np.full(367, np.nan)
    for d in range(1, 367):
        dist = np.minimum((doy_train - d) % 366, (d - doy_train) % 366)
        m = dist <= window
        probs[d] = float(y_train[m].mean()) if m.any() else base
    return probs


def predict_seasonal_climatology(
    doy_train: np.ndarray,
    y_train: np.ndarray,
    doy_test: np.ndarray,
    window: int = 15,
) -> np.ndarray:
    """baseline forecast สำหรับชุด test: ความน่าจะเป็นตามวันในปี (เรียนรู้จาก train เท่านั้น)."""
    probs = seasonal_climatology(doy_train, y_train, window)
    return probs[np.asarray(doy_test, dtype=int)]


def persistence_probs(
    state_train: np.ndarray, y_train: np.ndarray, state_test: np.ndarray
) -> np.ndarray:
    """baseline "persistence แบบมีเงื่อนไข": P(event ในสัปดาห์เป้าหมาย | สถานะ ณ วันออกพยากรณ์).

    state = 0/1 เช่น "วันออกพยากรณ์อยู่ในช่วง heatwave หรือไม่"
    ประมาณความน่าจะเป็นแบบมีเงื่อนไขจากชุด train แล้ว map ให้ชุด test
    (ยุติธรรมกว่า persistence ดิบที่พยากรณ์ 0/1 ตรงๆ ซึ่งโดน Brier ลงโทษหนัก)
    สถานะที่ไม่เคยพบใน train -> ใช้ base rate รวมของ train
    """
    state_train = np.asarray(state_train, dtype=int)
    y_train = np.asarray(y_train, dtype=float)
    state_test = np.asarray(state_test, dtype=int)
    base = float(y_train.mean()) if y_train.size else float("nan")
    p_by_state = {}
    for s in np.unique(state_train):
        m = state_train == s
        p_by_state[int(s)] = float(y_train[m].mean()) if m.any() else base
    return np.array([p_by_state.get(int(s), base) for s in state_test])


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


def evaluate_probabilistic(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline_prob: float | np.ndarray | None = None,
) -> dict:
    """รวมเมตริกหลักไว้ใน dict เดียว (ใช้บันทึกผลต่อ run).

    baseline_prob: ส่ง baseline จากชุด train เสมอเมื่อรายงานผลจริง (ดู brier_skill_score)
    """
    y_true = np.asarray(y_true)
    return {
        "n": int(y_true.size),
        "base_rate": float(np.mean(y_true)),
        "brier": brier(y_true, y_prob),
        "bss": brier_skill_score(y_true, y_prob, baseline_prob),
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

    # 5) BSS กับ baseline ภายนอก (จาก train): พยากรณ์ = baseline -> BSS = 0 พอดี
    train_rate = 0.12  # สมมุติ base rate จากชุด train (ต่างจาก test เล็กน้อย)
    same_as_baseline = evaluate_probabilistic(y, np.full(n, train_rate), baseline_prob=train_rate)
    assert abs(same_as_baseline["bss"]) < 1e-12, same_as_baseline
    # baseline จาก train (ไม่เห็นเฉลย test) ต้องแพ้ baseline ที่ใช้ base rate ของ test เอง
    bss_vs_train = brier_skill_score(y, np.full(n, y.mean()), baseline_prob=train_rate)
    assert bss_vs_train > 0, bss_vs_train
    print(f"BSS เทียบ baseline จาก train: พยากรณ์=baseline -> 0 ✅ | test-rate ชนะ train-rate (BSS={bss_vs_train:.4f}) ✅")

    # 6) seasonal climatology: y มีรอบฤดูกาล -> ต้องเรียนรู้ความถี่ตาม doy ได้
    n2 = 366 * 20
    doy = (np.arange(n2) % 366) + 1
    p_true = np.where((doy >= 60) & (doy <= 150), 0.30, 0.02)  # "ฤดูร้อน" เสี่ยงสูง
    y2 = (rng.random(n2) < p_true).astype(float)
    doy_q = np.array([100, 300])  # กลางฤดูร้อน / นอกฤดู
    p_hat = predict_seasonal_climatology(doy, y2, doy_q, window=15)
    assert abs(p_hat[0] - 0.30) < 0.05, p_hat
    assert abs(p_hat[1] - 0.02) < 0.05, p_hat
    # ต้องชนะ base rate คงที่ เมื่อใช้เป็น baseline ของกันและกัน
    p_seasonal_all = predict_seasonal_climatology(doy, y2, doy, window=15)
    bss_seasonal = brier_skill_score(y2, p_seasonal_all, baseline_prob=y2.mean())
    assert bss_seasonal > 0.1, bss_seasonal
    print(f"seasonal climatology: p̂(doy=100)={p_hat[0]:.3f}≈0.30, p̂(doy=300)={p_hat[1]:.3f}≈0.02, "
          f"ชนะ base rate คงที่ (BSS={bss_seasonal:.3f}) ✅")

    # 7) persistence แบบมีเงื่อนไข: y สหสัมพันธ์กับสถานะปัจจุบัน -> conditional prob ถูกต้อง
    state = (rng.random(n2) < 0.2).astype(int)
    p_cond = np.where(state == 1, 0.5, 0.05)
    y3 = (rng.random(n2) < p_cond).astype(float)
    p_pers = persistence_probs(state, y3, np.array([0, 1]))
    assert abs(p_pers[0] - 0.05) < 0.02, p_pers
    assert abs(p_pers[1] - 0.50) < 0.03, p_pers
    print(f"persistence มีเงื่อนไข: P(y|state=0)={p_pers[0]:.3f}≈0.05, P(y|state=1)={p_pers[1]:.3f}≈0.50 ✅")

    print("[OK] เมตริกทั้งหมดถูกต้องตามค่าที่ทราบ")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
