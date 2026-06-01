"""Capture the six UI screenshots that Chapter 17 of the project report references.

Spawns a headless Chromium via Playwright, navigates to the running Sleep Twin
app at http://127.0.0.1:8765, drives the UI into the specific states the
report describes, and saves the PNGs to artifacts/screenshots/.

Usage:
    1. start the app:  .venv/Scripts/python.exe scripts/run_app.py
    2. run this:       .venv/Scripts/python.exe scripts/capture_screenshots.py
"""

from __future__ import annotations

import io
import random
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "screenshots"
APP_URL = "http://127.0.0.1:8765"
VIEWPORT = {"width": 1440, "height": 900}


# ---------------------------------------------------------------------------
# Synthetic Apple Health export (for screenshot 6)
# ---------------------------------------------------------------------------

def _synth_apple_zip() -> bytes:
    tz = timezone(timedelta(hours=-7))
    anchor = datetime(2026, 5, 22, 23, 30, tzinfo=tz)
    records: list[str] = []

    def add(rtype: str, start: datetime, end: datetime, value: str, source: str = "Apple Watch"):
        s = start.strftime("%Y-%m-%d %H:%M:%S ") + start.strftime("%z")
        e = end.strftime("%Y-%m-%d %H:%M:%S ") + end.strftime("%z")
        records.append(
            f'<Record type="{rtype}" sourceName="{source}" '
            f'startDate="{s}" endDate="{e}" value="{value}" />'
        )

    random.seed(42)

    # 7 prior nights with detailed Apple stages
    for d in range(7, 0, -1):
        night_start = anchor - timedelta(days=d)
        t = night_start
        for cycle in range(5):
            ce = t + timedelta(minutes=60 + random.randint(-5, 5))
            add("HKCategoryTypeIdentifierSleepAnalysis", t, ce, "HKCategoryValueSleepAnalysisAsleepCore"); t = ce
            de = t + timedelta(minutes=15)
            add("HKCategoryTypeIdentifierSleepAnalysis", t, de, "HKCategoryValueSleepAnalysisAsleepDeep"); t = de
            re = t + timedelta(minutes=12)
            add("HKCategoryTypeIdentifierSleepAnalysis", t, re, "HKCategoryValueSleepAnalysisAsleepREM"); t = re
            if cycle < 4 and random.random() < 0.5:
                we = t + timedelta(minutes=2)
                add("HKCategoryTypeIdentifierSleepAnalysis", t, we, "HKCategoryValueSleepAnalysisAwake"); t = we

    # Target night
    t = anchor
    for cycle in range(5):
        ce = t + timedelta(minutes=60); add("HKCategoryTypeIdentifierSleepAnalysis", t, ce, "HKCategoryValueSleepAnalysisAsleepCore"); t = ce
        de = t + timedelta(minutes=15); add("HKCategoryTypeIdentifierSleepAnalysis", t, de, "HKCategoryValueSleepAnalysisAsleepDeep"); t = de
        re = t + timedelta(minutes=12); add("HKCategoryTypeIdentifierSleepAnalysis", t, re, "HKCategoryValueSleepAnalysisAsleepREM"); t = re
        if cycle < 4:
            we = t + timedelta(minutes=2); add("HKCategoryTypeIdentifierSleepAnalysis", t, we, "HKCategoryValueSleepAnalysisAwake"); t = we
    target_end = t

    # HR samples every 2 min across the 8-day window
    hr_t = anchor - timedelta(days=8)
    while hr_t < target_end + timedelta(hours=1):
        h = hr_t.hour
        if 23 <= h or h <= 7:
            bpm = 55 + random.gauss(0, 2.5)
        else:
            bpm = 72 + random.gauss(0, 4)
        add("HKQuantityTypeIdentifierHeartRate", hr_t, hr_t, f"{bpm:.1f}")
        hr_t += timedelta(seconds=120)

    # Real Apple HRV SDNN, resting HR, steps, active energy, exercise time
    for d in range(8):
        t = anchor - timedelta(days=d, hours=2)
        sdnn = 52 + random.gauss(0, 6)
        add("HKQuantityTypeIdentifierHeartRateVariabilitySDNN", t, t, f"{sdnn:.1f}")
    for d in range(8):
        t = anchor - timedelta(days=d, hours=6)
        rhr = 58 + random.gauss(0, 2)
        add("HKQuantityTypeIdentifierRestingHeartRate", t, t, f"{rhr:.1f}")
    for d in range(8, -1, -1):
        day_start = anchor.replace(hour=8, minute=0) - timedelta(days=d)
        total_steps = random.randint(6000, 12000)
        remaining = total_steps
        step_t = day_start
        while remaining > 0 and step_t < day_start + timedelta(hours=14):
            chunk = min(remaining, random.randint(50, 400))
            add("HKQuantityTypeIdentifierStepCount", step_t, step_t + timedelta(minutes=30), str(chunk))
            remaining -= chunk
            step_t += timedelta(minutes=45)
        add("HKQuantityTypeIdentifierActiveEnergyBurned", day_start, day_start + timedelta(hours=12), str(random.randint(400, 850)))
        add("HKQuantityTypeIdentifierAppleExerciseTime", day_start, day_start + timedelta(hours=12), str(random.randint(20, 60)))

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<HealthData>\n' + "\n".join(records) + "\n</HealthData>\n"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("apple_health_export/export.xml", xml)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Capture flow
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()

        page.goto(APP_URL)
        # Let fonts + initial analysis fetch settle
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1200)

        # --- 1. Landing view -------------------------------------------------
        print("capturing 01_landing.png …")
        page.screenshot(path=str(OUT / "01_landing.png"), full_page=True)

        # --- 3. Constellation with all overlays on (do this BEFORE the flip so PSG truth stays toggled) ---
        # PSG truth is off by default; click the chip to enable it.
        # HR and Motion are on by default; leave them.
        print("capturing 03_constellation_overlays.png …")
        psg_chip = page.locator('.chip-toggle[data-overlay="psg"]').first
        psg_chip.click()
        page.wait_for_timeout(400)
        constellation_panel = page.locator(".constellation-panel").first
        constellation_panel.screenshot(path=str(OUT / "03_constellation_overlays.png"))
        # Toggle PSG truth back off so it doesn't bleed into the next shots
        psg_chip.click()
        page.wait_for_timeout(300)

        # --- 2. Score-card flip overlay (Recovery) ---------------------------
        print("capturing 02_score_flip.png …")
        recovery_card = page.locator('.card[data-score="recovery"]').first
        recovery_card.click()
        page.wait_for_timeout(700)  # let the flip animation settle
        page.screenshot(path=str(OUT / "02_score_flip.png"), full_page=True)
        # close the overlay
        page.locator("#flip-close").click()
        page.wait_for_timeout(400)

        # --- 4. What-if dock with a 90-min sleep cut applied -----------------
        print("capturing 04_whatif.png …")
        slider = page.locator("#dial-sleep")
        # Fire input + change so the JS debounced handler runs
        slider.evaluate("(el) => { el.value = 90; el.dispatchEvent(new Event('input', {bubbles:true})); el.dispatchEvent(new Event('change', {bubbles:true})); }")
        page.wait_for_timeout(800)  # debounce + server roundtrip
        # Scroll the page so the whatif dock is on screen, then full-page screenshot
        page.locator(".whatif").scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "04_whatif.png"), full_page=True)
        # reset to 0 so future shots are clean
        slider.evaluate("(el) => { el.value = 0; el.dispatchEvent(new Event('input', {bubbles:true})); el.dispatchEvent(new Event('change', {bubbles:true})); }")
        page.wait_for_timeout(400)

        # --- 5. Methodology drawer ------------------------------------------
        print("capturing 05_methodology_drawer.png …")
        # scroll back to top so the drawer + aurora are visible together
        page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
        page.wait_for_timeout(300)
        page.keyboard.press("KeyM")
        page.wait_for_timeout(700)  # drawer slide-in
        # Scroll the drawer's body so we land on a section that shows several score headings
        page.evaluate("document.getElementById('drawer-body').scrollTop = 380")
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "05_methodology_drawer.png"), full_page=False)
        page.locator("#drawer-close").click()
        page.wait_for_timeout(400)

        # --- 6. Apple Health import + audit ---------------------------------
        print("capturing 06_apple_import_audit.png (uploads a synthetic export) …")
        page.locator("#open-import").click()
        page.wait_for_timeout(400)
        # Inject a synthetic Apple Health zip into the file input
        zip_bytes = _synth_apple_zip()
        page.locator("#import-file").set_input_files(
            files=[{"name": "export.zip", "mimeType": "application/zip", "buffer": zip_bytes}]
        )
        # The upload + parse + analysis takes a couple of seconds for this sized export.
        # The import modal auto-closes ~700 ms after a successful upload, so wait for
        # the overlay element to actually be hidden (Playwright's default waits for
        # visibility, hence the explicit state="hidden").
        page.wait_for_selector("#import-overlay", state="hidden", timeout=15000)
        page.wait_for_timeout(800)
        # Scroll down so the audit card is visible alongside the bento grid
        page.locator("#apple-insights").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "06_apple_import_audit.png"), full_page=True)

        browser.close()

    print()
    print(f"all six screenshots written to {OUT.relative_to(ROOT)}/")
    for name in sorted(p.name for p in OUT.iterdir() if p.suffix == ".png"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
