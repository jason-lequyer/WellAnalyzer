// ---------- Accurate detection (NO RoiManager) ----------
List<Roi> detectWells(FloatProcessor srcFP, double thr, boolean invert,
                      double minArea, double maxArea,
                      int keepTopN, double borderMarginPx,
                      boolean doMorphClose){
  int w = srcFP.getWidth(), h = srcFP.getHeight(), nPix = w*h

  // Threshold to binary
  ByteProcessor bin = new ByteProcessor(w, h)
  float[] src = (float[]) srcFP.getPixels()
  byte[] dst = (byte[]) bin.getPixels()
  if (!invert){
    for (int i=0; i<nPix; i++) dst[i] = (byte) ((src[i] >= thr) ? 255 : 0)
  } else {
    for (int i=0; i<nPix; i++) dst[i] = (byte) ((src[i] <= thr) ? 255 : 0)
  }

  // Optional morphology to close small gaps
  if (doMorphClose){
    BinaryProcessor bp = new BinaryProcessor(bin)
    bp.dilate(); bp.erode()
    bin = (ByteProcessor) bp
  }

  // Analyze particles WITHOUT adding to RoiManager
  ResultsTable rt = new ResultsTable()
  int meas = Measurements.AREA | Measurements.CENTROID | Measurements.ELLIPSE
  ParticleAnalyzer pa = new ParticleAnalyzer(ParticleAnalyzer.SHOW_NONE, meas, rt,
      Math.max(0, minArea), (maxArea>0 ? maxArea : Double.POSITIVE_INFINITY),
      0.0, 1.0)
  pa.setHideOutputImage(true)
  pa.analyze(new ImagePlus("bin", bin))

  // Build candidate circles from measurements
  int n = rt.size()
  if (n == 0) return []

  double[] area = new double[n], cx = new double[n], cy = new double[n], rEst = new double[n]
  for (int i=0; i<n; i++){
    double A  = rt.getValue("Area", i)
    double Xc = rt.getValue("X",    i)  // centroid X
    double Yc = rt.getValue("Y",    i)  // centroid Y
    double maj = rt.getColumnIndex("Major")>=0 ? rt.getValue("Major", i) : Double.NaN
    double min = rt.getColumnIndex("Minor")>=0 ? rt.getValue("Minor", i) : Double.NaN

    if (!Double.isFinite(A) || A <= 0) continue
    if (!Double.isFinite(Xc) || !Double.isFinite(Yc)) continue

    // Robust radius estimate: average of ellipse axes if available, else from area
    double r = (Double.isFinite(maj) && Double.isFinite(min) && maj>0 && min>0)
             ? 0.25*(maj + min)
             : Math.sqrt(A/Math.PI)

    // Border filter (center must be inside margins)
    if (Xc < borderMarginPx || Xc > w - borderMarginPx || Yc < borderMarginPx || Yc > h - borderMarginPx) continue

    area[i] = A; cx[i] = Xc; cy[i] = Yc; rEst[i] = r
  }

  // Indices of valid rows
  List<Integer> ids = new ArrayList<>()
  for (int i=0; i<n; i++) if (area[i] > 0) ids.add(i)
  if (ids.isEmpty()) return []

  // Sort by area (largest first)
  ids.sort({ a,b -> Double.compare(area[b], area[a]) } as Comparator)

  // Deduplicate close centers (anticipate repeated detections)
  // Use median radius to pick a gentle separation threshold.
  double[] radiiValid = new double[ids.size()]
  for (int k=0; k<ids.size(); k++) radiiValid[k] = rEst[ids.get(k)]
  Arrays.sort(radiiValid)
  double medR = (radiiValid.length==0) ? 0 : (radiiValid.length%2==1 ? radiiValid[radiiValid.length/2]
                                                                     : 0.5*(radiiValid[radiiValid.length/2-1] + radiiValid[radiiValid.length/2]))
  double minCenterDist = 0.3 * (2.0 * Math.max(1e-9, medR))

  List<Roi> circles = new ArrayList<>()
  List<double[]> centers = new ArrayList<>()
  for (int k=0; k<ids.size(); k++){
    int i = ids.get(k)
    double x = cx[i], y = cy[i], r = rEst[i]
    boolean tooClose=false
    for (double[] c : centers){
      if (Math.hypot(c[0]-x, c[1]-y) < minCenterDist){ tooClose=true; break }
    }
    if (tooClose) continue
    circles.add(new OvalRoi(x - r, y - r, 2*r, 2*r))
    centers.add([x,y] as double[])
    if (keepTopN > 0 && circles.size() >= keepTopN) break
  }

  return circles
}
