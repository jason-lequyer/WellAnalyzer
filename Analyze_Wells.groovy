/* WellAnalyzer.groovy – Scripted plugin (no compilation required)
 * Usage: Open an image, then Run from the Script Editor (Language: Groovy).
 * Optional: save to Fiji.app/plugins/Well Analyzer/WellAnalyzer.groovy, then Help ▸ Refresh Menus.
 */

import ij.IJ
import ij.ImagePlus
import ij.gui.*
import ij.process.*
import ij.plugin.*
import ij.plugin.filter.*
import ij.plugin.frame.*
import ij.measure.*
import ij.util.Tools

// ===== Swing UI =====
import javax.swing.JDialog
import javax.swing.JPanel
import javax.swing.JLabel
import javax.swing.JSlider
import javax.swing.JCheckBox
import javax.swing.JSpinner
import javax.swing.SpinnerNumberModel
import javax.swing.JButton
import javax.swing.BorderFactory
import javax.swing.Box
import javax.swing.BoxLayout

// ===== AWT / Swing helpers =====
import java.awt.Font
import java.awt.Dimension
import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.GridBagLayout
import java.awt.GridBagConstraints
import java.awt.Insets
import java.awt.Color
import java.awt.Rectangle
import java.awt.RenderingHints
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.event.ActionListener
import javax.swing.event.ChangeListener

// ===== Util =====
import java.util.List
import java.util.ArrayList
import java.util.Arrays
import java.util.Comparator

// ----------------- Small helpers -----------------
double clamp(double v, double lo, double hi){ Math.max(lo, Math.min(hi, v)) }
double otsuIntensity(ImageProcessor ip, double imgMin, double imgMax){
  int[] h = ip.getHistogram()
  int t = new AutoThresholder().getThreshold(AutoThresholder.Method.Otsu, h)
  int bins = Math.max(1, h.length - 1)
  imgMin + (t / (double)bins) * (imgMax - imgMin)
}
double median(double[] a){
  if (a==null || a.length==0) return 0
  double[] b = a.clone(); Arrays.sort(b)
  int n = b.length
  if (n == 1) return b[0]
  int mid = n.intdiv(2)
  return (n % 2 == 1) ? b[mid] : 0.5 * (b[mid - 1] + b[mid])
}
ImagePlus toGrayscaleIfNeeded(ImagePlus src){
  if (src==null) return null
  if (src.getType()==ImagePlus.GRAY8 || src.getType()==ImagePlus.GRAY16 || src.getType()==ImagePlus.GRAY32) return src
  ImagePlus dup = src.duplicate()
  IJ.run(dup, "8-bit", "")
  return dup
}

// -------------- Circle-of-best-fit (Kåsa) --------------
double[] fitCircleKasa(float[] xs, float[] ys, int n){
  if (n < 3) return null
  double Sx=0, Sy=0, Sxx=0, Syy=0, Sxy=0, Sx3=0, Sy3=0, Sxxy=0, Sxyy=0
  for (int i=0; i<n; i++){
    double x = xs[i], y = ys[i]
    double x2 = x*x, y2 = y*y
    Sx += x; Sy += y
    Sxx += x2; Syy += y2; Sxy += x*y
    Sx3 += x2*x; Sy3 += y2*y
    Sxxy += x2*y; Sxyy += x*y2
  }
  double A11=Sxx, A12=Sxy, A13=Sx
  double A21=Sxy, A22=Syy, A23=Sy
  double A31=Sx,  A32=Sy,  A33=n
  double B1=-(Sx3+Sxyy), B2=-(Sxxy+Sy3), B3=-(Sxx+Syy)

  double detA =
    A11*(A22*A33 - A23*A32) -
    A12*(A21*A33 - A23*A31) +
    A13*(A21*A32 - A22*A31)
  if (Math.abs(detA) < 1e-12) return null

  double detA1 =
    B1*(A22*A33 - A23*A32) -
    A12*(B2*A33 - A23*B3) +
    A13*(B2*A32 - A22*B3)
  double detA2 =
    A11*(B2*A33 - A23*B3) -
    B1*(A21*A33 - A23*A31) +
    A13*(A21*B3 - B2*A31)
  double detA3 =
    A11*(A22*B3 - B2*A32) -
    A12*(A21*B3 - B2*A31) +
    B1*(A21*A32 - A22*A31)

  double a = detA1/detA
  double b = detA2/detA
  double c = detA3/detA

  double cx = -a/2.0
  double cy = -b/2.0
  double r2 = cx*cx + cy*cy - c
  if (!Double.isFinite(r2) || r2 <= 0) return null
  double r = Math.sqrt(r2)
  return [cx, cy, r] as double[]
}
Roi bestFitCircleRoi(Roi r, int imgW, int imgH){
  FloatPolygon fp = r.getFloatPolygon()
  if (fp==null || fp.npoints < 3){
    Rectangle b = r.getBounds()
    double rad = 0.5 * Math.min(b.width, b.height)
    return new OvalRoi(b.x + b.width/2.0 - rad, b.y + b.height/2.0 - rad, 2*rad, 2*rad)
  }
  double[] fit = fitCircleKasa(fp.xpoints, fp.ypoints, fp.npoints)
  if (fit==null){
    Rectangle b = r.getBounds()
    double rad = 0.5 * Math.min(b.width, b.height)
    return new OvalRoi(b.x + b.width/2.0 - rad, b.y + b.height/2.0 - rad, 2*rad, 2*rad)
  }
  double cx=fit[0], cy=fit[1], rad=fit[2]
  rad = Math.max(2.0, Math.min(rad, Math.min(Math.min(cx, imgW-cx), Math.min(cy, imgH-cy))))
  return new OvalRoi(cx - rad, cy - rad, 2*rad, 2*rad)
}

// ---------------- Overlays & masks ----------------
void overlayCirclesWithNumbers(ImagePlus imp, List<Roi> rois, Color circleColor, Color labelColor, Font font){
  Overlay ov = new Overlay()
  int idx = 1
  for (Roi r : rois){
    Roi rr = (Roi) r.clone()
    rr.setStrokeColor(circleColor)
    rr.setStrokeWidth(2.5f)
    rr.setFillColor(null)
    ov.add(rr)

    Rectangle b = r.getBounds()
    double cx = b.x + b.width/2.0
    double cy = b.y + b.height/2.0

    TextRoi tr
    try {
      tr = new TextRoi((int)Math.round(cx-6), (int)Math.round(cy-6), String.valueOf(idx))
      try { tr.setCurrentFont(font) } catch(Throwable t2) { tr.setFont(font.getName(), font.getStyle(), font.getSize()) }
    } catch(Throwable t){
      tr = new TextRoi(cx-6, cy-6, String.valueOf(idx), font)
    }
    tr.setStrokeColor(labelColor)
    tr.setAntiAlias(true)
    ov.add(tr)
    idx++
  }
  imp.setOverlay(ov)
  imp.updateAndDraw()
}

int[] labelMapFromRois(List<Roi> rois, int w, int h){
  int[] labels = new int[w*h]
  Arrays.fill(labels, 0)
  for (int idx=0; idx<rois.size(); idx++){
    Roi r = rois.get(idx)
    Rectangle b = r.getBounds()
    ByteProcessor m = (ByteProcessor) r.getMask()
    if (m != null){
      byte[] mp = (byte[]) m.getPixels()
      for (int yy=0; yy<b.height; yy++){
        int rowOff = (b.y + yy) * w
        int mOff = yy * b.width
        for (int xx=0; xx<b.width; xx++){
          if ((mp[mOff + xx] & 0xff) != 0){
            int i = rowOff + (b.x + xx)
            if (i>=0 && i<labels.length && labels[i]==0) labels[i] = idx+1
          }
        }
      }
    } else {
      double rx = b.width/2.0, ry = b.height/2.0
      double cx = b.x + rx,   cy = b.y + ry
      for (int yy=0; yy<b.height; yy++){
        int rowOff = (b.y + yy) * w
        double y = (b.y + yy + 0.5 - cy)/ry
        double y2 = y*y
        for (int xx=0; xx<b.width; xx++){
          double x = (b.x + xx + 0.5 - cx)/rx
          if (x*x + y2 <= 1.0){
            int i = rowOff + (b.x + xx)
            if (i>=0 && i<labels.length && labels[i]==0) labels[i] = idx+1
          }
        }
      }
    }
  }
  return labels
}

FloatProcessor maskFloatInside(int w, int h, float[] src, boolean[] inside){
  FloatProcessor out = new FloatProcessor(w, h, src.clone())
  float[] p = (float[]) out.getPixels()
  int n = p.length
  for (int i=0; i<n; i++) if (!inside[i]) p[i]=0f
  return out
}

int[] computeCropBounds(boolean[] inside, int w, int h, int pad){
  int minx=w, miny=h, maxx=-1, maxy=-1
  for (int y=0; y<h; y++){
    int off = y*w
    for (int x=0; x<w; x++){
      if (inside[off+x]){
        if (x<minx) minx=x
        if (x>maxx) maxx=x
        if (y<miny) miny=y
        if (y>maxy) maxy=y
      }
    }
  }
  if (maxx<0){ return [0,0,w,h] as int[] }
  int x0 = Math.max(0, minx - pad)
  int y0 = Math.max(0, miny - pad)
  int x1 = Math.min(w-1, maxx + pad)
  int y1 = Math.min(h-1, maxy + pad)
  return [x0, y0, (x1-x0+1), (y1-y0+1)] as int[]
}
boolean[] cropMask(boolean[] mask, int w, int h, int x0, int y0, int cw, int ch){
  boolean[] out = new boolean[cw*ch]
  for (int y=0; y<ch; y++){
    int sy = y0 + y
    int so = sy*w + x0
    int doff = y*cw
    for (int x=0; x<cw; x++){
      out[doff+x] = mask[so+x]
    }
  }
  return out
}
List<Roi> shiftRois(List<Roi> rois, int dx, int dy){
  List<Roi> out = new ArrayList<>()
  for (Roi r : rois){
    Roi rr = (Roi) r.clone()
    Rectangle b = rr.getBounds()
    rr.setLocation(b.x - dx, b.y - dy)
    out.add(rr)
  }
  return out
}

// Build base grayscale RGB (from a wells-only FloatProcessor)
int[] buildBaseRGB(FloatProcessor srcFP, double imgMin, double imgMax){
  int w = srcFP.getWidth(), h = srcFP.getHeight(), n = w*h
  float[] src = (float[]) srcFP.getPixels()
  int[] base = new int[n]
  double scale = (imgMax > imgMin) ? (255.0/(imgMax - imgMin)) : 1.0
  for (int i=0; i<n; i++){
    int g = (int) Math.round(clamp((src[i]-imgMin)*scale, 0, 255))
    base[i] = (0xFF<<24) | (g<<16) | (g<<8) | g
  }
  return base
}
ByteProcessor buildSelectedMask(FloatProcessor srcFP, boolean[] insideMask, double thr){
  int w = srcFP.getWidth(), h = srcFP.getHeight(), n = w*h
  float[] src = (float[]) srcFP.getPixels()
  ByteProcessor bp = new ByteProcessor(w, h)
  byte[] dst = (byte[]) bp.getPixels()
  for (int i=0; i<n; i++) dst[i] = (byte)((insideMask[i] && src[i]>=thr) ? 255 : 0)
  return bp
}

// --- POP TINT overlay (revised):
// Selected pixels: magenta tint + slight dim of underlying gray.
// Non-selected INSIDE wells: brightened grayscale for contrast.
// Outside wells: unchanged (already black in the wells-only image).
void applyPopTintOverlay(ImagePlus targetImp, int[] baseRGB, FloatProcessor srcFP,
                         boolean[] insideMask, double thr,
                         int tintR, int tintG, int tintB, double tintAlpha,
                         double dimUnderTintFactor, double brightenFactor,
                         List<Roi> circleRois, Font labelFont) {
  int w = srcFP.getWidth(), h = srcFP.getHeight(), n = w*h
  float[] src = (float[]) srcFP.getPixels()
  int[] out = Arrays.copyOf(baseRGB, baseRGB.length)

  int rT = tintR & 0xff, gT = tintG & 0xff, bT = tintB & 0xff
  double ia = 1.0 - tintAlpha

  for (int i=0; i<n; i++){
    int px = out[i]
    int g = px & 0xff
    if (insideMask[i]){
      if (src[i] >= thr){
        // Dim the base gray slightly under the tint
        int gDim = (int)Math.round(g * dimUnderTintFactor)
        int r2 = (int)Math.round(ia*gDim + tintAlpha*rT)
        int g2 = (int)Math.round(ia*gDim + tintAlpha*gT)
        int b2 = (int)Math.round(ia*gDim + tintAlpha*bT)
        out[i] = (0xFF<<24) | ((r2&0xff)<<16) | ((g2&0xff)<<8) | (b2&0xff)
      } else {
        // Brighten the non-selected pixels inside wells
        int gB = (int)Math.round(Math.min(255, g * brightenFactor))
        out[i] = (0xFF<<24) | ((gB&0xff)<<16) | ((gB&0xff)<<8) | (gB&0xff)
      }
    } else {
      // outside wells: keep as-is (black)
      out[i] = px
    }
  }

  // Rebuild overlay each time: image + red circles + white numbers
  ColorProcessor cp = new ColorProcessor(w, h, out)
  ImageRoi imgLayer = new ImageRoi(0, 0, cp); imgLayer.setOpacity(1.0)
  Overlay ov = new Overlay(); ov.add(imgLayer)

  int idx = 1
  for (Roi r : circleRois){
    Roi rr = (Roi) r.clone()
    rr.setStrokeColor(Color.red); rr.setStrokeWidth(2.5f); rr.setFillColor(null); ov.add(rr)
    Rectangle b = rr.getBounds()
    double cx = b.x + b.width/2.0, cy = b.y + b.height/2.0
    TextRoi tr
    try { tr = new TextRoi((int)Math.round(cx-6), (int)Math.round(cy-6), String.valueOf(idx)); tr.setCurrentFont(labelFont) }
    catch(Throwable t){ tr = new TextRoi(cx-6, cy-6, String.valueOf(idx), labelFont) }
    tr.setStrokeColor(Color.white); tr.setAntiAlias(true); ov.add(tr)
    idx++
  }

  targetImp.setOverlay(ov); targetImp.updateAndDraw()
}

// ---------- Histogram panel (masked) ----------
class HistPanel extends JPanel {
  int[] hist
  double thr
  double imgMin, imgMax
  HistPanel(int[] h, double t, double lo, double hi){ hist=h; thr=t; imgMin=lo; imgMax=hi; setPreferredSize(new Dimension(320,120)) }
  void setThr(double t){ thr=t; repaint() }
  void setHist(int[] h){ hist=h; repaint() }
  @Override
  protected void paintComponent(Graphics g){
    super.paintComponent(g)
    Graphics2D g2 = (Graphics2D) g
    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    int W = getWidth(), H = getHeight()
    g2.setColor(new Color(245,245,245)); g2.fillRect(0,0,W,H)
    g2.setColor(new Color(200,200,200)); g2.drawRect(0,0,W-1,H-1)

    if (hist==null || hist.length==0) return
    int bins = hist.length
    int maxC = 1; for (int v: hist) if (v>maxC) maxC=v

    int x0=32, y0=H-20, ww=W-42, hh=H-30
    g2.setColor(new Color(220,220,220))
    g2.drawLine(x0, y0, x0+ww, y0)
    g2.drawLine(x0, y0-hh, x0, y0)
    g2.setColor(new Color(90,90,90))
    int px=-1, py=-1
    for (int i=0; i<bins; i++){
      double xf = i/(double)(bins-1)
      int x = x0 + (int)Math.round(xf*(ww-1))
      int y = y0 - (int)Math.round(hh*(hist[i]/(double)maxC))
      if (px>=0) g2.drawLine(px,py,x,y)
      px=x; py=y
    }
    double tf = (thr - imgMin) / Math.max(1e-9, (imgMax - imgMin))
    if (tf < 0) tf = 0; else if (tf > 1) tf = 1
    int xT = x0 + (int)Math.round(tf*(ww-1))
    g2.setColor(Color.RED)
    g2.drawLine(xT, y0-hh, xT, y0)
  }
}

// Masked histogram of wells-only FP (bins=256)
int[] maskedHistogram(FloatProcessor fp, boolean[] inside, double lo, double hi, int bins){
  int n = fp.getWidth()*fp.getHeight()
  float[] p = (float[]) fp.getPixels()
  int[] h = new int[bins]
  double scale = (bins-1)/Math.max(1e-9, (hi-lo))
  for (int i=0; i<n; i++){
    if (!inside[i]) continue
    double v = p[i]
    int b = (int)Math.floor((v - lo)*scale + 0.5)
    if (b<0) b=0; else if (b>=bins) b=bins-1
    h[b]++
  }
  return h
}

// --------------- Core detection ----------------
List<Roi> detectWells(FloatProcessor srcFP, double thr, boolean invert,
                      double minArea, double maxArea,
                      int keepTopN, double minSpacingPx, double borderMarginPx,
                      boolean doMorphClose){
  int w = srcFP.getWidth(), h = srcFP.getHeight(), nPix = w*h
  ByteProcessor bin = new ByteProcessor(w, h)
  float[] src = (float[]) srcFP.getPixels()
  byte[] dst = (byte[]) bin.getPixels()
  if (!invert){
    for (int i=0; i<nPix; i++) dst[i] = (byte) ((src[i] >= thr) ? 255 : 0)
  } else {
    for (int i=0; i<nPix; i++) dst[i] = (byte) ((src[i] <= thr) ? 255 : 0)
  }
  if (doMorphClose){
    BinaryProcessor bp = new BinaryProcessor(bin)
    bp.dilate(); bp.erode()
    bin = (ByteProcessor) bp
  }

  ImagePlus binImp = new ImagePlus("bin", bin)
  ResultsTable rt = new ResultsTable()
  RoiManager rm = RoiManager.getInstance2()
  if (rm==null) rm = new RoiManager()

  int options = ParticleAnalyzer.ADD_TO_MANAGER
  int meas = Measurements.AREA | Measurements.CENTROID | Measurements.ELLIPSE
  ParticleAnalyzer pa = new ParticleAnalyzer(options, meas, rt,
      Math.max(0, minArea), (maxArea>0 ? maxArea : Double.POSITIVE_INFINITY),
      0.0, 1.0)
  pa.setHideOutputImage(true)
  try { pa.setRoiManager(rm) } catch(Throwable ignore) {}
  pa.analyze(binImp)

  Roi[] rois = rm.getRoisAsArray()
  try { rm.runCommand("Show None") } catch(Throwable ignore) {}
  try { rm.reset() } catch(Throwable ignore) {}
  try { rm.setVisible(false) } catch(Throwable ignore) {}
  if (rois==null || rois.length==0) return []

  double[] areas = new double[rois.length], cx = new double[rois.length], cy = new double[rois.length]
  for (int i=0; i<rois.length; i++){
    binImp.setRoi(rois[i])
    def s = binImp.getStatistics(Measurements.AREA | Measurements.CENTROID)
    areas[i]=s.area; cx[i]=s.xCentroid; cy[i]=s.yCentroid
  }

  boolean[] ok = new boolean[rois.length]
  int nOK = 0
  for (int i=0; i<rois.length; i++){
    boolean inside = (cx[i] >= borderMarginPx) && (cx[i] <= w - borderMarginPx) &&
                     (cy[i] >= borderMarginPx) && (cy[i] <= h - borderMarginPx)
    ok[i] = inside
    if (inside) nOK++
  }
  if (nOK == 0) return []

  Integer[] idx = new Integer[nOK]; int cnt=0
  for (int i=0; i<rois.length; i++) if (ok[i]) idx[cnt++] = i
  Arrays.sort(idx, { Integer i1, Integer i2 -> Double.compare(areas[i2], areas[i1]) } as Comparator)

  double[] okAreas = new double[nOK]; int j=0
  for (int i=0; i<rois.length; i++) if (ok[i]) okAreas[j++] = areas[i]
  double medArea = median(okAreas)
  double medRadius = Math.sqrt(Math.max(1e-9, medArea) / Math.PI)
  double minCenterDist = 0.3 * (2.0 * medRadius)

  List<Roi> kept = new ArrayList<>()
  List<double[]> centers = new ArrayList<>()
  for (int k=0; k<idx.length; k++){
    int i = idx[k]
    double x=cx[i], y=cy[i]
    boolean tooClose=false
    for (double[] c : centers){
      if (Math.hypot(c[0]-x, c[1]-y) < minCenterDist){ tooClose=true; break }
    }
    if (!tooClose){
      kept.add(rois[i])
      centers.add([x,y] as double[])
      if (keepTopN > 0 && kept.size() >= keepTopN) break
    }
  }
  return kept
}

// ---------------- Measurements ----------------
void computeAndShowFinalResults(ImagePlus workImp, int[] labelMap, int nWells, double thr2){
  FloatProcessor src = (FloatProcessor) workImp.getProcessor()
  float[] p = (float[]) src.getPixels(); int nPix = p.length
  long[] countAll = new long[nWells+1], countSel = new long[nWells+1]
  double[] sumAll = new double[nWells+1], sumSel = new double[nWells+1]

  for (int i=0; i<nPix; i++){
    int label = (labelMap==null)?0:labelMap[i]
    if (label==0) continue
    double v = p[i]
    countAll[label]++; sumAll[label]+=v
    if (v>=thr2){ countSel[label]++; sumSel[label]+=v }
  }

  ResultsTable rt = new ResultsTable()
  for (int w=1; w<=nWells; w++){
    double areaPct = (countAll[w]==0)?0:(100.0*countSel[w]/(double)countAll[w])
    double intPct  = (sumAll[w]==0)?0:(100.0*sumSel[w]/sumAll[w])
    rt.incrementCounter()
    rt.addValue("Well #", w)
    rt.addValue("Area Percent", Math.round(areaPct*100.0)/100.0)
    rt.addValue("Intensity Percent", Math.round(intPct*100.0)/100.0)
  }
  rt.show("Well Analysis")
  IJ.showStatus("Well Analyzer: Done")
}

// ==================== MAIN ====================
ImagePlus imp = IJ.getImage()
if (imp==null){ IJ.error("Open an image first."); return }
ImagePlus gray = toGrayscaleIfNeeded(imp)
ImagePlus workImp = gray.duplicate()
if (workImp.getBitDepth()!=32) workImp.setProcessor(workImp.getProcessor().convertToFloatProcessor())

int W = workImp.getWidth(), H = workImp.getHeight()
def st = workImp.getProcessor().getStatistics()
double imgMin = st.min, imgMax = st.max
if (!Double.isFinite(imgMin) || !Double.isFinite(imgMax) || imgMin==imgMax){ imgMin=0; imgMax=255 }

// Precompute (speed)
FloatProcessor fpRaw = (FloatProcessor) workImp.getProcessor().duplicate()
FloatProcessor fpDenoised = (FloatProcessor) fpRaw.duplicate()
new RankFilters().rank(fpDenoised, 2.0, RankFilters.MEDIAN)

// Slider mapping
def toSlider   = { double t -> (int)Math.round(1000.0 * ((t - imgMin) / Math.max(1e-9, (imgMax - imgMin)))) }
def fromSlider = { int sv     -> imgMin + (sv/1000.0) * (imgMax - imgMin) }

// ==== STEP 1: Single slider + numbered red circles ====
class Step1Config {
  double thr
  double minArea
  double maxArea
  int    keepTopN
  double borderMargin
  int    cropPad
  boolean invert
  boolean morphClose
  boolean useMedian
  boolean circleFit
  boolean accepted = false
}
Step1Config cfg1 = new Step1Config()
cfg1.thr         = otsuIntensity(workImp.getProcessor(), imgMin, imgMax)
cfg1.minArea     = 2500
cfg1.maxArea     = 0
cfg1.keepTopN    = 6
cfg1.borderMargin= 10
cfg1.cropPad     = 8
cfg1.invert      = false
cfg1.morphClose  = true
cfg1.useMedian   = true
cfg1.circleFit   = true

JDialog d1 = new JDialog((java.awt.Frame)null, "Well Analyzer – Step 1 (Detect Wells)", true)
d1.setLayout(new BorderLayout(10,10))
JPanel top = new JPanel(); top.setLayout(new BoxLayout(top, BoxLayout.Y_AXIS))
JLabel title = new JLabel("Detect wells – numbered red circles (live)")
title.setFont(new Font("SansSerif", Font.BOLD, 16)); title.setAlignmentX(JLabel.CENTER_ALIGNMENT)
top.add(title); top.add(Box.createVerticalStrut(6))
JPanel thrPanel = new JPanel(new BorderLayout(6,6))
thrPanel.setAlignmentX(JPanel.CENTER_ALIGNMENT)
JLabel thrLabel = new JLabel(String.format("Well threshold: %.2f", cfg1.thr), JLabel.CENTER)
thrLabel.setFont(new Font("SansSerif", Font.PLAIN, 14))
JSlider thrSlider = new JSlider(0,1000, toSlider(cfg1.thr))
thrSlider.setMajorTickSpacing(250); thrSlider.setMinorTickSpacing(50)
thrSlider.setPaintTicks(true); thrSlider.setPaintLabels(true)
thrSlider.setPreferredSize(new Dimension(520, 54))
thrPanel.add(thrLabel, BorderLayout.NORTH); thrPanel.add(thrSlider, BorderLayout.CENTER)
top.add(thrPanel)

// Advanced
JPanel adv = new JPanel(new GridBagLayout())
adv.setBorder(BorderFactory.createTitledBorder("Advanced"))
GridBagConstraints ga = new GridBagConstraints()
ga.insets=new Insets(3,3,3,3); ga.fill=GridBagConstraints.HORIZONTAL; ga.weightx=1.0
JSpinner spMinArea = new JSpinner(new SpinnerNumberModel(cfg1.minArea as Double, 0.0d as Double, 1.0e12d as Double, 100.0d as Double))
JSpinner spMaxArea = new JSpinner(new SpinnerNumberModel(cfg1.maxArea as Double, 0.0d as Double, 1.0e12d as Double, 1000.0d as Double))
JSpinner spKeepN   = new JSpinner(new SpinnerNumberModel(Integer.valueOf(cfg1.keepTopN), Integer.valueOf(0), Integer.valueOf(9999), Integer.valueOf(1)))
JSpinner spBorder  = new JSpinner(new SpinnerNumberModel(cfg1.borderMargin as Double, 0.0d as Double, 1.0e6d as Double, 1.0d as Double))
JSpinner spCropPad = new JSpinner(new SpinnerNumberModel(Integer.valueOf(cfg1.cropPad), Integer.valueOf(0), Integer.valueOf(200), Integer.valueOf(1)))
JCheckBox cbInvert = new JCheckBox("Wells are dark on bright background", cfg1.invert)
JCheckBox cbClose  = new JCheckBox("Morphological close (dilate+erode)", cfg1.morphClose)
JCheckBox cbMedian = new JCheckBox("Use median denoise (precomputed)", cfg1.useMedian)
JCheckBox cbCircle = new JCheckBox("Use circle-of-best-fit masks", cfg1.circleFit)
String[] labels = ["Min area (px²)", "Max area (px², 0=∞)", "Keep top N wells (0=all)", "Border margin (px)", "Crop border (px)"]
def comps = [spMinArea, spMaxArea, spKeepN, spBorder, spCropPad]
for (int i=0; i<labels.length; i++){ ga.gridx=0; ga.gridy=i; adv.add(new JLabel(labels[i]), ga); ga.gridx=1; adv.add(comps[i], ga) }
ga.gridx=0; ga.gridy=labels.length; ga.gridwidth=2; adv.add(cbInvert, ga)
ga.gridy++; adv.add(cbClose, ga)
ga.gridy++; adv.add(cbMedian, ga)
ga.gridy++; adv.add(cbCircle, ga)
top.add(Box.createVerticalStrut(8)); top.add(adv)
d1.add(top, BorderLayout.CENTER)
JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT)); JButton ok1 = new JButton("OK"); JButton cancel1 = new JButton("Cancel")
buttons.add(cancel1); buttons.add(ok1); d1.add(buttons, BorderLayout.SOUTH)
d1.pack(); d1.setLocationRelativeTo(null)

// Live preview
long lastUpdate = 0L
final long MIN_INTERVAL_MS = 60L
List<Roi> wellRois = []
Font labelFont = new Font("SansSerif", Font.BOLD, 13)
def doDetectAndPreview = {
  long now = System.currentTimeMillis()
  if (now - lastUpdate < MIN_INTERVAL_MS) return
  lastUpdate = now
  cfg1.minArea     = ((Number)spMinArea.getValue()).doubleValue()
  cfg1.maxArea     = ((Number)spMaxArea.getValue()).doubleValue()
  cfg1.keepTopN    = ((Number)spKeepN.getValue()).intValue()
  cfg1.borderMargin= ((Number)spBorder.getValue()).doubleValue()
  cfg1.cropPad     = ((Number)spCropPad.getValue()).intValue()
  cfg1.invert      = cbInvert.isSelected()
  cfg1.morphClose  = cbClose.isSelected()
  cfg1.useMedian   = cbMedian.isSelected()
  cfg1.circleFit   = cbCircle.isSelected()

  FloatProcessor srcFP = cfg1.useMedian ? fpDenoised : fpRaw
  List<Roi> rois = detectWells(srcFP, cfg1.thr, cfg1.invert,
      cfg1.minArea, cfg1.maxArea,
      cfg1.keepTopN, 0, cfg1.borderMargin,
      cfg1.morphClose)

  List<Roi> circles = new ArrayList<>()
  if (cfg1.circleFit){ for (Roi r : rois) circles.add(bestFitCircleRoi(r, W, H)) } else { circles = rois }
  try { RoiManager rm = RoiManager.getInstance2(); if (rm!=null){ rm.runCommand("Show None"); rm.reset(); rm.setVisible(false) } } catch(Throwable ignore) {}
  overlayCirclesWithNumbers(imp, circles, Color.red, Color.white, labelFont)
  wellRois = circles
}
thrSlider.addChangeListener({ e -> cfg1.thr = fromSlider(thrSlider.getValue()); thrLabel.setText(String.format("Well threshold: %.2f", cfg1.thr)); doDetectAndPreview() } as ChangeListener)
thrLabel.setText(String.format("Well threshold: %.2f", cfg1.thr))
doDetectAndPreview()
ok1.addActionListener({ e -> cfg1.accepted = true; d1.dispose() } as ActionListener)
cancel1.addActionListener({ e -> cfg1.accepted = false; d1.dispose() } as ActionListener)
d1.setVisible(true)
if (!cfg1.accepted){ imp.setOverlay(null); return }
if (wellRois.isEmpty()){ IJ.error("No wells detected. Try lowering Min area, or enabling 'Wells are dark'."); imp.setOverlay(null); return }

// ==== Build mask & wells-only cropped image ====
int[] labelMap = labelMapFromRois(wellRois, W, H)
boolean[] insideMask = new boolean[W*H]; for (int i=0; i<insideMask.length; i++) insideMask[i] = (labelMap[i] != 0)
int[] cb = computeCropBounds(insideMask, W, H, cfg1.cropPad)
int x0 = cb[0], y0 = cb[1], CW = cb[2], CH = cb[3]

// Copy only inside pixels into crop
float[] srcAll = (float[])((FloatProcessor)workImp.getProcessor()).getPixels()
float[] dstCrop = new float[CW*CH]
for (int y=0; y<CH; y++){
  int sy = y0 + y
  for (int x=0; x<CW; x++){
    int sx = x0 + x
    int si = sy*W + sx
    int di = y*CW + x
    dstCrop[di] = insideMask[si] ? srcAll[si] : 0f
  }
}
FloatProcessor wellsFP = new FloatProcessor(CW, CH, dstCrop)
ImagePlus wellsImp = new ImagePlus(imp.getTitle()+" [wells-only,cropped]", wellsFP)
wellsImp.show()

// Shift circles and number them on the cropped image
List<Roi> shifted = shiftRois(wellRois, x0, y0)
overlayCirclesWithNumbers(wellsImp, shifted, Color.red, Color.white, labelFont)

// --- Zoom to 100% exactly ---
try { IJ.run(wellsImp, "Original Scale", "") } catch(Throwable ignore) {}
// (Optional) make the window a bit larger without changing zoom
try {
  def win = wellsImp.getWindow()
  if (win!=null) win.setSize(new Dimension(Math.max(720, win.getWidth()+160), Math.max(560, win.getHeight()+120)))
} catch(Throwable ignore) {}
imp.setOverlay(null); imp.updateAndDraw()

// ==== STEP 2: Tint-only (pop) + Histogram + Live stats ====
class Step2Config { double thr; boolean accepted=false }
Step2Config cfg2 = new Step2Config()
cfg2.thr = otsuIntensity(wellsImp.getProcessor(), imgMin, imgMax)

boolean[] insideCrop = cropMask(insideMask, W, H, x0, y0, CW, CH)
int[] baseRGB  = buildBaseRGB(wellsFP, imgMin, imgMax)
int[] hist = maskedHistogram(wellsFP, insideCrop, imgMin, imgMax, 256)
HistPanel histPanel = new HistPanel(hist, cfg2.thr, imgMin, imgMax)

JDialog d2 = new JDialog((java.awt.Frame)null, "Well Analyzer – Step 2 (Contents)", true)
d2.setLayout(new BorderLayout(10,10))
JPanel t2 = new JPanel(); t2.setLayout(new BoxLayout(t2, BoxLayout.Y_AXIS))
JLabel title2 = new JLabel("Contents threshold – vivid tint inside circles (outside is black)")
title2.setFont(new Font("SansSerif", Font.BOLD, 16)); title2.setAlignmentX(JLabel.CENTER_ALIGNMENT)
t2.add(title2); t2.add(Box.createVerticalStrut(6))

JPanel thr2Panel = new JPanel(new BorderLayout(6,6))
thr2Panel.setAlignmentX(JPanel.CENTER_ALIGNMENT)
JLabel thr2Label = new JLabel(String.format("Content threshold: %.2f", cfg2.thr), JLabel.CENTER)
thr2Label.setFont(new Font("SansSerif", Font.PLAIN, 14))
JSlider thr2Slider = new JSlider(0, 1000, toSlider(cfg2.thr))
thr2Slider.setMajorTickSpacing(250); thr2Slider.setMinorTickSpacing(50)
thr2Slider.setPaintTicks(true); thr2Slider.setPaintLabels(true)
thr2Slider.setPreferredSize(new Dimension(520, 54))
thr2Panel.add(thr2Label, BorderLayout.NORTH); thr2Panel.add(thr2Slider, BorderLayout.CENTER)
t2.add(thr2Panel)

// Histogram + live stats
JPanel statsPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 12, 4))
JLabel lblArea = new JLabel("Area selected: -- %")
JLabel lblInt  = new JLabel("Intensity selected: -- %")
statsPanel.add(histPanel)
JPanel statsText = new JPanel(); statsText.setLayout(new BoxLayout(statsText, BoxLayout.Y_AXIS))
statsText.add(lblArea); statsText.add(Box.createVerticalStrut(6)); statsText.add(lblInt)
statsPanel.add(statsText)
t2.add(Box.createVerticalStrut(6)); t2.add(statsPanel)

d2.add(t2, BorderLayout.CENTER)
JPanel btn2 = new JPanel(new FlowLayout(FlowLayout.RIGHT))
JButton cancel2 = new JButton("Cancel"); JButton ok2 = new JButton("OK")
btn2.add(cancel2); btn2.add(ok2); d2.add(btn2, BorderLayout.SOUTH)
d2.pack(); d2.setLocationRelativeTo(null)

// Live stats
def updateLiveStats = { ->
  float[] p = (float[]) wellsFP.getPixels()
  long cntAll=0, cntSel=0; double sumAll=0, sumSel=0
  int n = p.length
  for (int i=0; i<n; i++){
    if (!insideCrop[i]) continue
    double v = p[i]; cntAll++; sumAll+=v
    if (v>=cfg2.thr){ cntSel++; sumSel+=v }
  }
  double areaPct = (cntAll==0)?0:(100.0*cntSel/cntAll)
  double intPct  = (sumAll==0)?0:(100.0*sumSel/sumAll)
  lblArea.setText(String.format("Area selected: %.2f%%", areaPct))
  lblInt.setText(String.format("Intensity selected: %.2f%%", intPct))
}

// Visualization updater (tint only, with strong emphasis)
final int TINT_R = 255, TINT_G = 64, TINT_B = 160    // vivid magenta
final double TINT_ALPHA = 0.60                        // tint opacity
final double DIM_UNDER_TINT = 0.70                    // dim grayscale under tint
final double BRIGHTEN_FACTOR = 1.25                   // brighten non-selected pixels inside wells

def refreshPreview = { ->
  histPanel.setThr(cfg2.thr)
  applyPopTintOverlay(wellsImp, baseRGB, wellsFP, insideCrop, cfg2.thr,
                      TINT_R, TINT_G, TINT_B, TINT_ALPHA,
                      DIM_UNDER_TINT, BRIGHTEN_FACTOR,
                      shifted, labelFont)
  updateLiveStats()
}

// initial apply
refreshPreview()

// Throttled updates
long last2 = 0L
final long MIN2 = 50L
thr2Slider.addChangeListener({ e ->
  long now = System.currentTimeMillis()
  if (now - last2 < MIN2) return
  last2 = now
  cfg2.thr = fromSlider(thr2Slider.getValue())
  thr2Label.setText(String.format("Content threshold: %.2f", cfg2.thr))
  refreshPreview()
} as ChangeListener)

ok2.addActionListener({ e -> cfg2.accepted = true; d2.dispose() } as ActionListener)
cancel2.addActionListener({ e -> cfg2.accepted = false; d2.dispose() } as ActionListener)

d2.setVisible(true)
if (!cfg2.accepted){
  overlayCirclesWithNumbers(wellsImp, shifted, Color.red, Color.white, labelFont)
  return
}

// Final compute (use original float image + original label map)
computeAndShowFinalResults(workImp, labelMap, wellRois.size(), cfg2.thr)
