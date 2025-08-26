
1. **Download/clone** this repo.
2. **Copy** the inner `plugins/Well_Analyzer/` folder into your Fiji installation:
   - **macOS**: `/Applications/Fiji.app/plugins/`
   - **Windows**: `C:\Fiji.app\plugins\`
   - **Linux**: `~/Fiji.app/plugins/`
3. Restart Fiji or run **Help ▸ Refresh Menus**.

You should now see the plugin at:

> **Plugins ▸ Well Analyzer ▸ Analyze Wells**

> _Note:_ Fiji replaces underscores with spaces in menu names (`Well_Analyzer` → `Well Analyzer`, `Analyze_Wells` → `Analyze Wells`). No compilation is required; Groovy comes with Fiji.

---

## Usage

1. **Open your image** (RGB will be converted to 8‑bit).
2. **Plugins ▸ Well Analyzer ▸ Analyze Wells**
3. **Step 1 – Detect wells**
   - Adjust the **Well threshold** slider until the red **best‑fit circles** match your wells.
   - Click **OK** to lock them in.
4. **Step 2 – Analyze contents**
   - A **wells‑only** (cropped) image opens at **100% zoom** (outside circles is black).
   - Move the **Content threshold** slider:
     - **Selected** pixels are tinted **magenta** and the gray underneath is slightly **dimmed**.
     - **Non‑selected** pixels **inside** the circles are **brightened**.
   - A **histogram** (inside wells only) and **Area% / Intensity%** readouts update live.
   - Click **OK** to produce the **ResultsTable**.

---

## Notes & defaults

- Step‑1 well detection uses sensible defaults internally:
  - **Min area:** 10,000 px² (≈100×100 pixels)
  - **Morphological close:** on (dilate + erode)
  - **Median denoise:** on (radius ≈ 2 px)
  - **Keep top N:** all (no limit)
  - Automatic choice of **bright wells vs dark wells** polarity (whichever yields more wells).
- Step‑2 visualization constants (edit in `Analyze_Wells.groovy` near the bottom):
  - `TINT_R, TINT_G, TINT_B` (tint color – default magenta)
  - `TINT_ALPHA` (tint opacity)
  - `DIM_UNDER_TINT` (how much to dim the grayscale under the tint)
  - `BRIGHTEN_FACTOR` (how much to brighten non‑selected pixels inside wells)

---

## Output

- A standard ImageJ **ResultsTable** with:
  - `Well #`
  - `Area Percent` — % of pixels inside the well that are ≥ the content threshold.
  - `Intensity Percent` — % of total integrated intensity inside the well coming from pixels ≥ the threshold.

You can **Save As…** CSV from the Results window.

---

## Troubleshooting

- **No wells detected?** Try moving the Step‑1 threshold slider; the plugin also auto‑tests inverted polarity and picks whichever returns more wells.
- **Slow slider updates?** Try smaller images for exploration; then run on full size for final numbers.

---

## License

MIT © Your Name (year)
