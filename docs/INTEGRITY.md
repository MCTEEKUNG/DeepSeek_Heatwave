# Integrity Note — leakage audit (2026-06-13)

ก่อน "ลงหลัก" ที่โปรเจกต์นี้ มีการตรวจและแก้ leakage 2 จุด ดูดีไซน์เต็มที่
`docs/superpowers/specs/2026-06-13-rigor-serving-hardening-design.md`.

## R2 — in_hw_today มองอนาคต (แก้แล้ว: eliminate)
`flag_heatwaves` นับ run แบบ fwd+bwd (ถูกต้องสำหรับ label) แต่ output เดิมถูกใช้เป็น
feature `in_hw_today` และ state ของ persistence baseline → ใช้ข้อมูลอนาคต ณ วันออกพยากรณ์
และทำให้ค่าวันล่าสุดตอน serve ไม่ตรง. แก้เป็น trailing-only (`trailing_run_length`).

ผลต่อสกิล (pooled BSS, y_rm, prod model Logistic (bal+cal)):

> หมายเหตุ provenance: "ก่อน" = snapshot ที่บันทึกก่อน R2 fix แต่หลัง MJO-impute fix (commit 15241ea,
> 2026-06-13 11:34) — ตัวเลขจึงต่ำกว่าตาราง README ที่รอบแรกสุด (0.194) เพราะ MJO fix ลดสกิลไปแล้วส่วนหนึ่ง.
> Baseline persistence ก็ถูก snapshot ณ จุดเดียวกัน และในชุด "หลัง" persistence BSS ลดลง (เช่น lead 2:
> 0.044 → 0.023) เพราะ in_hw_today state ที่ clean ไม่มี look-ahead อีกต่อไป — คาดได้ ไม่ใช่ regression.

| lead | BSS ก่อน (pre-R2) | BSS หลัง (trailing) | beats clim | beats persist |
| --- | --- | --- | --- | --- |
| 2 | 0.152 | 0.153 [+0.057, +0.244] | ✓ | ✓ |
| 3 | 0.103 | 0.106 [+0.016, +0.176] | ✓ | ✓ |
| 4 | 0.130 | 0.135 [+0.052, +0.196] | ✓ | ✓ |
| 5 | 0.133 | 0.134 [+0.067, +0.187] | ✓ | ✓ |
| 6 | 0.112 | 0.112 [+0.046, +0.171] | ✓ | ✓ |

> สรุปสั้น: หลังแก้ leak R2 โมเดล Logistic (bal+cal) ยังชนะทั้ง climatology และ persistence อย่างมีนัยสำคัญ (BSS 95% CI ไม่คร่อม 0, q_vs_persist < 0.05 หลัง BH-FDR) ครบทุก lead 2–6. ค่า BSS ของโมเดลแทบไม่เปลี่ยน (Δ ≤ 0.005) สอดคล้องกับ permutation importance ที่ in_hw_today มี mean_drop_bss ≈ 0 — leak นี้มีผลน้อยมากต่อตัวเลขรายงาน แต่ยังต้องแก้เพื่อความถูกต้องตอน serve.

## R1 — percentile-label leak (gate: ดู Task 4)
เกณฑ์ p90 ที่นิยาม label คำนวณบน 30 ปีเต็มรวม test. วัด ΔBSS แบบ leave-block-out:
ΔBSS ที่วัดได้ = <เติมจาก leak_check_r1.py — Task 4> → <อยู่ใน / หลุด> 95% CI ของ results_master
→ การตัดสิน: <document พอ (คง frozen-all-history) / ทำ per-fold elimination>
