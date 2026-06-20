from __future__ import annotations

THRESHOLD_NOTE = (
    'ℹ️ "สูง" = โอกาสเกิดมากกว่าปกติ ≥ 2.5 เท่า, '
    '"ค่อนข้างสูง" = ≥ 1.5 เท่า'
)


def build_weekly_summary(selection: dict, website_url: str, max_list: int = 12) -> str:
    issue = selection.get("issue_date", "")
    high = [r["name_th"] for r in selection.get("high", [])]
    elevated = [r["name_th"] for r in selection.get("elevated", [])]
    warnings = selection.get("warnings", [])
    total = len(high) + len(elevated)

    if total == 0:
        msg = (
            f"✅ พยากรณ์สัปดาห์นี้ (ออก {issue}) ไม่มีจังหวัดที่ความเสี่ยงคลื่นความร้อน"
            f"สูงผิดปกติในอีก 2–4 สัปดาห์\n👉 ดูรายละเอียดทุกจังหวัด: {website_url}"
        )
        return _append_warnings(msg, warnings)

    disp_high = high[:max_list]
    remaining = max_list - len(disp_high)
    disp_elev = elevated[: max(0, remaining)]
    extra = total - len(disp_high) - len(disp_elev)

    lines = [
        f"🌡️ เฝ้าระวังคลื่นความร้อน — พยากรณ์ออก {issue}",
        "จังหวัดที่มีความเสี่ยงต่อสุขภาพในอีก 2–4 สัปดาห์:",
    ]
    if disp_high:
        lines.append("🔴 สูง: " + ", ".join(disp_high))
    if disp_elev:
        lines.append("🟠 ค่อนข้างสูง: " + ", ".join(disp_elev))
    if extra > 0:
        lines.append(f"…และอีก {extra} จังหวัด — ดูทั้งหมดบนเว็บ")
    lines.append(THRESHOLD_NOTE)
    lines.append(f"👉 ดูแผนที่ภาพรวม: {website_url}")
    return _append_warnings("\n".join(lines), warnings)


def build_about_message(website_url: str) -> str:
    return (
        "ℹ️ วิธีอ่านระดับความเสี่ยงคลื่นความร้อน\n"
        "ระบบเทียบ 'โอกาสเกิด' กับค่าปกติของแต่ละจังหวัด:\n"
        "• ปกติ = ใกล้ค่าเฉลี่ย\n"
        "• ค่อนข้างสูง = มากกว่าปกติ ≥ 1.5 เท่า\n"
        "• สูง = มากกว่าปกติ ≥ 2.5 เท่า\n"
        "พยากรณ์ล่วงหน้า 2–4 สัปดาห์ (sub-seasonal) เป็นการ 'บ่งชี้ความเสี่ยง' "
        "ไม่ใช่การฟันธงว่าจะเกิดแน่นอน\n"
        f"👉 แผนที่ภาพรวมทุกจังหวัด: {website_url}"
    )


def build_welcome_message(website_url: str) -> str:
    return (
        "สวัสดีครับ 👋 นี่คือบริการแจ้งเตือนความเสี่ยงคลื่นความร้อนรายสัปดาห์ของไทย\n"
        "• แตะ 'ความเสี่ยงสัปดาห์นี้' เพื่อดูจังหวัดเสี่ยง\n"
        "• แตะ 'ดูแผนที่ภาพรวม' เพื่อเปิดเว็บไซต์\n"
        f"👉 {website_url}"
    )


def _append_warnings(msg: str, warnings) -> str:
    if warnings:
        msg += "\n\n⚠️ หมายเหตุ: " + " ".join(warnings)
    return msg
