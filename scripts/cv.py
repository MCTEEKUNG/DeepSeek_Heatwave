"""
Blocked / rolling-origin time-series cross-validation (กัน temporal leakage)

ทำไมต้องมี: ข้อมูลอนุกรมเวลามี autocorrelation สูง + เราทำนายล่วงหน้า 2-6 สัปดาห์
ถ้าใช้ random CV โมเดลจะ "เห็นอนาคต" ผ่านวันที่อยู่ติดกัน -> สกิลเฟ้อ (leakage)

วิธีแก้: แบ่งตามเวลาเป็นบล็อก, train อยู่ "ก่อน" test เสมอ, และเว้น "ช่องว่าง (gap/embargo)"
ระหว่างปลาย train กับต้น test อย่างน้อยเท่ากับ (lead time + ความยาวหน้าต่างเป้าหมาย)
เพื่อตัดการรั่วจากความสัมพันธ์ข้ามช่วงเวลา
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass
class RollingOriginCV:
    """สร้างชุด (train_idx, test_idx) แบบ rolling-origin มี gap.

    พารามิเตอร์ (หน่วย = จำนวน time steps เช่น 'วัน'):
      n_splits   : จำนวน fold
      test_size  : ความยาวบล็อก test แต่ละ fold
      gap        : embargo ระหว่างปลาย train กับต้น test (>= lead + target window)
      expanding  : True = train ขยายตัวสะสม, False = train ขนาดคงที่ (sliding)
      train_size : ใช้เมื่อ expanding=False (จำนวน steps ของ train window)
    """
    n_splits: int = 5
    test_size: int = 28          # ~4 สัปดาห์
    gap: int = 42                # ~6 สัปดาห์ (ครอบ lead สูงสุด) เพื่อกัน leakage
    expanding: bool = True
    train_size: int | None = None

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """yield (train_idx, test_idx) โดย index อ้างถึงตำแหน่งบนแกนเวลาที่ 'เรียงแล้ว'."""
        need = self.n_splits * self.test_size + self.gap + 1
        if n_samples < need:
            raise ValueError(
                f"ข้อมูลสั้นเกินไป: ต้องการอย่างน้อย ~{need} steps แต่มี {n_samples}"
            )

        # วาง test block ต่อท้ายกัน ไล่จากท้ายสุดของอนุกรมเวลาถอยขึ้นมา
        test_starts = [
            n_samples - (k + 1) * self.test_size for k in reversed(range(self.n_splits))
        ]

        for test_start in test_starts:
            test_end = test_start + self.test_size
            train_end = test_start - self.gap  # เว้น gap ก่อนเริ่ม test
            if train_end <= 0:
                continue

            if self.expanding or self.train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.train_size)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx


def _selftest() -> None:
    cv = RollingOriginCV(n_splits=4, test_size=28, gap=42, expanding=True)
    n = 365 * 5  # 5 ปีจำลอง
    folds = list(cv.split(n))
    assert len(folds) == 4, f"ควรได้ 4 folds ได้ {len(folds)}"

    for i, (tr, te) in enumerate(folds, 1):
        # 1) train ต้องอยู่ก่อน test ทั้งหมด
        assert tr.max() < te.min(), "train ต้องมาก่อน test"
        # 2) ต้องมี gap >= ที่กำหนด ระหว่างปลาย train กับต้น test
        actual_gap = te.min() - tr.max() - 1
        assert actual_gap >= cv.gap, f"fold {i}: gap={actual_gap} < {cv.gap}"
        # 3) ไม่มี index ซ้ำกันระหว่าง train/test
        assert len(np.intersect1d(tr, te)) == 0, "train/test ห้ามทับกัน"
        print(f"fold {i}: train [{tr.min():4d}..{tr.max():4d}] "
              f"gap={actual_gap} test [{te.min():4d}..{te.max():4d}]")

    print("[OK] ทุก fold: train มาก่อน test, มี gap เพียงพอ, ไม่ทับกัน")

    # ทดสอบโหมด sliding (train คงที่)
    cv2 = RollingOriginCV(n_splits=3, test_size=20, gap=10, expanding=False, train_size=100)
    for tr, te in cv2.split(500):
        assert len(tr) <= 100
    print("[OK] โหมด sliding: train_size ถูกจำกัดตามกำหนด")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
