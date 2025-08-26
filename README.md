
1. **Download/clone** this repo.
2. **Copy** the inner `plugins/Well_Analyzer/` folder into your Fiji installation:
   - **macOS**: `/Applications/Fiji.app/plugins/`
   - **Windows**: `C:\Fiji.app\plugins\`
   - **Linux**: `~/Fiji.app/plugins/`
3. Restart Fiji or run **Help ▸ Refresh Menus**.

You should now see the plugin at:

> **Plugins ▸ Well Analyzer ▸ Analyze Wells**


---

## Usage

1. **Open your image**
2. **Plugins ▸ Well Analyzer ▸ Analyze Wells**
3. **Step 1 – Detect wells**
   - Adjust the **Well threshold** slider until the red **best‑fit circles** match your wells.
   - Click **OK** to lock them in.
4. **Step 2 – Analyze contents**
   - A **wells‑only** (cropped) image opens
   - Move the **Content threshold** slider to segment, **Selected** pixels are tinted **magenta**.
   - Click **OK** to produce the **ResultsTable**.


---

## Output

- A standard ImageJ **ResultsTable** with:
  - `Well #`
  - `Area Percent` — % of pixels inside the well that are ≥ the content threshold.
  - `Intensity Percent` — % of total integrated intensity inside the well coming from pixels ≥ the threshold.

You can **Save As…** CSV from the Results window.

