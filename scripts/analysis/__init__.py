"""ชุดวิเคราะห์ robustness ของโมเดลพยากรณ์ heatwave (เฟส A: เสริมความแกร่ง + วิเคราะห์ผล).

โมดูลย่อย:
  io_utils  - โหลด predictions + ENSO regime + ค่าคงที่กลาง (REPORT_MODELS, paths, ฯลฯ)
  stats     - เครื่องยนต์สถิติ numpy ล้วน: moving-block bootstrap, paired test,
              Brier decomposition (Murphy), ECE, Benjamini–Hochberg FDR
  bootstrap_ci, regime_strat, calibration, permutation_importance, ablation,
  figures, make_report  - การวิเคราะห์ราย "เฟส" (ดู docs/แผน)

หลักการ: ส่วนใหญ่ "อ่านอย่างเดียว" จาก outputs/predictions.csv — ไม่เทรนใหม่/ไม่โหลด ERA5 ใหม่
ผลลัพธ์ทั้งหมดลง outputs/analysis/ (ไม่ทับของเดิม)
"""
