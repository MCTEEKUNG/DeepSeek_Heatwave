"""
เครื่องยนต์สถิติสำหรับงานวิเคราะห์ robustness — numpy ล้วน (ไม่พึ่ง scipy/statsmodels)

ทำไมต้อง "block" bootstrap:
  คำพยากรณ์เป็นอนุกรมเวลา autocorrelation สูง (หน้าต่างเป้าหมาย 7 วันซ้อนกัน + ฟีเจอร์เปลี่ยนช้า)
  ถ้า resample แบบ i.i.d. (สุ่มทีละวัน) จะ "ทำลาย" autocorrelation -> CI แคบเกินจริง
  -> สรุปว่ามีนัยสำคัญทั้งที่อาจไม่ใช่. moving-block bootstrap (MBB) สุ่ม "บล็อกของวันที่ติดกัน"
  ยาว L วัน เพื่อคงโครงสร้าง autocorrelation ภายในบล็อก -> CI สมจริง

ผูกกับเมตริกกลางใน evaluate.py (reuse reliability_curve, brier) เพื่อให้สอดคล้องทั้งโปรเจกต์
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import reliability_curve, brier  # noqa: E402  (reuse เมตริกกลาง)

DEFAULT_BLOCK = 28   # วัน (~4 สัปดาห์) ผูกกับโครงสร้างเป้าหมาย sub-seasonal
DEFAULT_B = 2000     # จำนวน bootstrap replicate
SEED = 42            # ตรงกับ models.RANDOM_STATE เพื่อ reproducibility


# --------------------------------------------------------- block bootstrap

def _block_indices(n: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """ดัชนี resample แบบ moving-block: สุ่มจุดเริ่มบล็อก (ทับซ้อนได้) ต่อกันให้ยาว >= n แล้วตัดเหลือ n."""
    L = max(1, min(int(L), n))
    n_blocks = int(np.ceil(n / L))
    max_start = n - L
    if max_start > 0:
        starts = rng.integers(0, max_start + 1, size=n_blocks)
    else:
        starts = np.zeros(n_blocks, dtype=int)
    return (starts[:, None] + np.arange(L)[None, :]).reshape(-1)[:n]


def moving_block_bootstrap(stat_fn, arrays, L=DEFAULT_BLOCK, B=DEFAULT_B, seed=SEED) -> np.ndarray:
    """คืน array ยาว B ของสถิติบนตัวอย่าง resample (อาจมี nan เช่น AUC ที่ resample ได้คลาสเดียว).

    arrays : tuple ของ 1D array ความยาวเท่ากัน
    สำคัญ  : resample "ดัชนีชุดเดียว" ใช้กับทุก array พร้อมกัน — y, p, p_clim ต้องเลื่อนด้วยกัน
             (ห้าม resample ตัวตั้ง/ตัวหารของ BSS แยกกัน ไม่งั้นค่าเพี้ยน)
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    out = np.empty(B, dtype=float)
    for b in range(B):
        idx = _block_indices(n, L, rng)
        out[b] = stat_fn(*[a[idx] for a in arrays])
    return out


def bootstrap_ci(stat_fn, arrays, L=DEFAULT_BLOCK, B=DEFAULT_B, seed=SEED, alpha=0.05) -> dict:
    """CI แบบ percentile จาก MBB. คืน point (ทั้งตัวอย่าง), lo, hi, n_valid, n_dropped(nan)."""
    point = float(stat_fn(*[np.asarray(a) for a in arrays]))
    boot = moving_block_bootstrap(stat_fn, arrays, L=L, B=B, seed=seed)
    valid = boot[~np.isnan(boot)]
    if valid.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_valid": 0, "n_dropped": int(B)}
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n_valid": int(valid.size), "n_dropped": int(B - valid.size)}


def paired_block_test(d, L=DEFAULT_BLOCK, B=DEFAULT_B, seed=SEED, alpha=0.05) -> dict:
    """ทดสอบ paired แบบ block-bootstrap บนผลต่างราย "วัน" d = bs_ref - bs_model.

    d > 0 = โมเดลดีกว่า baseline (Brier ต่ำกว่า). คืน:
      mean_d, ci_lo/hi, se, n,
      p_boot   : p ด้านเดียว (เปอร์เซ็นไทล์) สำหรับ H0: mean_d <= 0  = (1+#{mean*<=0})/(B+1)
      p_normal : normal-approx ผ่าน math.erf — ไว้ cross-check (ไม่พึ่ง scipy)
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)
    for b in range(B):
        means[b] = d[_block_indices(n, L, rng)].mean()
    mean_d = float(d.mean())
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = float(means.std(ddof=1))
    p_boot = float((1 + np.sum(means <= 0)) / (B + 1))
    if se > 0:
        z = mean_d / se
        p_normal = float(0.5 * (1 - math.erf(z / math.sqrt(2))))  # = 1 - Phi(z)
    else:
        p_normal = 0.0 if mean_d > 0 else 1.0
    return {"mean_d": mean_d, "ci_lo": float(lo), "ci_hi": float(hi),
            "se": se, "p_boot": p_boot, "p_normal": p_normal, "n": int(n)}


# ----------------------------------------------- calibration diagnostics

def brier_decomposition(y, p, n_bins: int = 10) -> dict:
    """แยก Brier ตาม Murphy:  brier = REL - RES + UNC + WBV

      REL (reliability, ต่ำ=ดี)  : ความน่าจะเป็นที่ทำนายห่างความถี่จริงแค่ไหน (calibration)
      RES (resolution, สูง=ดี)   : แยกแยะ event/non-event ได้แค่ไหน
      UNC (uncertainty)          : ความยากในตัวปัญหา = ō(1-ō)
      WBV (within-bin variance>=0): ส่วนต่างจากการแบ่ง bin ทำให้ identity ปิดพอดี

    (identity REL-RES+UNC = "binned Brier" เป๊ะ ; WBV = brier จริง - binned Brier)
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    N = y.size
    obar = float(y.mean())
    mp, of, ct = reliability_curve(y, p, n_bins=n_bins)
    ok = ct > 0
    REL = float(np.sum(ct[ok] * (mp[ok] - of[ok]) ** 2) / N)
    RES = float(np.sum(ct[ok] * (of[ok] - obar) ** 2) / N)
    UNC = float(obar * (1.0 - obar))
    bs = float(brier(y, p))
    WBV = float(bs - (REL - RES + UNC))
    return {"REL": REL, "RES": RES, "UNC": UNC, "WBV": WBV, "brier": bs}


def ece(y, p, n_bins: int = 10) -> float:
    """Expected Calibration Error = ค่าเฉลี่ยถ่วงจำนวน ของ |forecast - observed| ต่อ bin (ต่ำ=ดี)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mp, of, ct = reliability_curve(y, p, n_bins=n_bins)
    ok = ct > 0
    N = ct.sum()
    return float(np.sum(ct[ok] * np.abs(mp[ok] - of[ok])) / N) if N else float("nan")


def mce(y, p, n_bins: int = 10) -> float:
    """Maximum Calibration Error = ช่องว่าง |forecast - observed| สูงสุดข้าม bin."""
    mp, of, ct = reliability_curve(y, p, n_bins=n_bins)
    ok = ct > 0
    return float(np.max(np.abs(mp[ok] - of[ok]))) if ok.any() else float("nan")


# ------------------------------------------------ multiple comparisons

def bh_fdr(pvals) -> np.ndarray:
    """Benjamini–Hochberg q-values (คุม false discovery rate). คง nan ไว้ตามเดิม."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    q = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]   # บังคับ monotonic จากท้าย
    res = np.empty(m)
    res[order] = np.clip(q, 0.0, 1.0)
    out[mask] = res
    return out


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    rng = np.random.default_rng(0)

    # 1) Brier decomposition: identity ปิดเสมอ + WBV=0 เมื่อ forecast แยก bin ชัด
    y = (rng.random(5000) < 0.3).astype(float)
    p = np.clip(0.3 + 0.2 * rng.standard_normal(5000), 0.01, 0.99)
    dec = brier_decomposition(y, p)
    assert abs(dec["brier"] - (dec["REL"] - dec["RES"] + dec["UNC"] + dec["WBV"])) < 1e-9, dec
    assert dec["WBV"] >= -1e-12, dec
    p3 = np.where(y == 1, 0.95, 0.05)
    p3[:50] = 0.55                      # 3 ค่าแยกคนละ bin -> WBV = 0
    assert abs(brier_decomposition(y, p3)["WBV"]) < 1e-9
    print("[OK] Brier decomposition: identity ปิด (WBV>=0) ; forecast แยก bin ชัด -> WBV=0")

    # 2) block bootstrap: CI ครอบค่าจริง + บน AR(1) ต้อง "กว้างกว่า" i.i.d.
    mean_fn = lambda a: float(a.mean())
    width = lambda c: c["hi"] - c["lo"]
    x = rng.standard_normal(800)        # iid mean 0
    ci = bootstrap_ci(mean_fn, (x,), L=1, B=1000)
    assert ci["lo"] < 0 < ci["hi"], ci
    ar = np.empty(800)                  # AR(1) autocorrelation สูง
    e = rng.standard_normal(800)
    ar[0] = e[0]
    for t in range(1, 800):
        ar[t] = 0.85 * ar[t - 1] + e[t]
    w_iid = width(bootstrap_ci(mean_fn, (ar,), L=1, B=1000))
    w_blk = width(bootstrap_ci(mean_fn, (ar,), L=40, B=1000))
    assert w_blk > w_iid, (w_blk, w_iid)
    print(f"[OK] block CI กว้างกว่า i.i.d. บน AR(1): {w_blk:.3f} > {w_iid:.3f} (เหตุผลที่ต้อง block)")

    # 3) paired test: ผลต่างบวกชัด -> p เล็ก ; ศูนย์ -> p ~ 0.5
    r_pos = paired_block_test(0.02 + 0.05 * rng.standard_normal(600), L=20, B=1000)
    assert r_pos["mean_d"] > 0 and r_pos["p_boot"] < 0.05, r_pos
    r_zero = paired_block_test(0.05 * rng.standard_normal(600), L=20, B=1000)
    assert 0.2 < r_zero["p_boot"] < 0.8, r_zero
    print(f"[OK] paired test: บวกชัด p={r_pos['p_boot']:.3f}(<0.05) ; ศูนย์ p={r_zero['p_boot']:.2f}(~0.5)")

    # 4) BH-FDR: q >= p, monotonic, รองรับ nan
    pv = np.array([0.001, 0.2, 0.01, 0.04])
    q = bh_fdr(pv)
    assert np.all(q >= pv - 1e-12), (q, pv)
    qn = bh_fdr([0.001, np.nan, 0.01])
    assert np.isnan(qn[1]) and not np.isnan(qn[0]), qn
    print(f"[OK] BH-FDR: q={np.round(q, 4)} (>=p, nan ผ่าน)")

    # 5) ECE: calibrate ดี -> ต่ำ ; เฟ้อ -> สูง
    yc = (rng.random(4000) < 0.2).astype(float)
    ece_good, ece_bad = ece(yc, np.full(4000, 0.2)), ece(yc, np.full(4000, 0.6))
    assert ece_good < 0.05 < ece_bad, (ece_good, ece_bad)
    print(f"[OK] ECE: calibrate ดี={ece_good:.3f} < เฟ้อ={ece_bad:.3f}")

    print("[OK] stats self-test ผ่านทั้งหมด")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
