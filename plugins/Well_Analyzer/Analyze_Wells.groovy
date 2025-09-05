// Rigid 3×2 Well Grid — Lightning Auto‑Propose (before UI) + Stage‑2 Contents (Mean Intensity)
// Fiji (ImageJ2 1.54f, Java 8). No UI/ROIs shown until initial auto‑propose completes.

import ij.IJ
import ij.ImagePlus
import ij.gui.*
import ij.process.*
import ij.measure.*

import javax.swing.*
import javax.swing.event.*
import java.awt.*
import java.awt.event.*
import java.util.concurrent.*
import groovy.transform.CompileStatic

// ---------------- Speed profile ----------------
@CompileStatic
class SpeedProfile {
  int MAX_SIDE, XSTEP_C, YSTEP_C, XSTEP_F, YSTEP_F, FINAL_XSTEP, FINAL_YSTEP
  double ANG_STEP0, ANG_SPAN, COARSE_STRIDE_MUL
  boolean EARLY_EXIT
  SpeedProfile(int MAX_SIDE,int XSTEP_C,int YSTEP_C,int XSTEP_F,int YSTEP_F,int FINAL_XSTEP,int FINAL_YSTEP,
               double ANG_STEP0,double ANG_SPAN,double COARSE_STRIDE_MUL,boolean EARLY_EXIT){
    this.MAX_SIDE=MAX_SIDE; this.XSTEP_C=XSTEP_C; this.YSTEP_C=YSTEP_C; this.XSTEP_F=XSTEP_F; this.YSTEP_F=YSTEP_F
    this.FINAL_XSTEP=FINAL_XSTEP; this.FINAL_YSTEP=FINAL_YSTEP
    this.ANG_STEP0=ANG_STEP0; this.ANG_SPAN=ANG_SPAN; this.COARSE_STRIDE_MUL=COARSE_STRIDE_MUL; this.EARLY_EXIT=EARLY_EXIT
  }
}
final SpeedProfile PROF_LIGHT = new SpeedProfile(360, 5,5, 3,3, 2,2, 3.0d, 8.0d, 1.00d, true)

// ---------------- Geometry & scoring ----------------
@CompileStatic
class GridOpt {
  static final int COLS = 3, ROWS = 2, N = COLS*ROWS

  static int[] buildHalfWidths(int r){
    int[] hw = new int[2*r+1]
    for (int dy=-r; dy<=r; dy++){
      int idx = dy + r
      int rem = r*r - dy*dy
      if (rem < 0) rem = 0
      hw[idx] = (int)Math.floor(Math.sqrt((double)rem))
    }
    return hw
  }

  static void rotatedOffsets(double dx, double dy, double angleDeg, double[] offX, double[] offY){
    double th = Math.toRadians(angleDeg)
    double ct = Math.cos(th), st = Math.sin(th); int k=0
    for (int rr=0; rr<ROWS; rr++){
      for (int cc=0; cc<COLS; cc++){
        double ox = cc*dx, oy = rr*dy
        offX[k] = ct*ox - st*oy
        offY[k] = st*ox + ct*oy
        k++
      }
    }
  }

  static void feasibleAnchorBounds(int W,int H,double[] offX,double[] offY,int rInt,double[] b){
    double minOffX=Double.POSITIVE_INFINITY,maxOffX=Double.NEGATIVE_INFINITY
    double minOffY=Double.POSITIVE_INFINITY,maxOffY=Double.NEGATIVE_INFINITY
    for (int i=0;i<N;i++){
      double ox=offX[i], oy=offY[i]
      if (ox<minOffX) minOffX=ox; if (ox>maxOffX) maxOffX=ox
      if (oy<minOffY) minOffY=oy; if (oy>maxOffY) maxOffY=oy
    }
    double r=(double)rInt
    b[0]=r - minOffX; b[1]=W - r - maxOffX; b[2]=r - minOffY; b[3]=H - r - maxOffY
  }

  /** Sum of row‑wise min across 6 circles (sampled) with bounds/safe indexing. */
  static double scoreMinProjSampledBound(
    float[] p, int W, int H,
    int[] hw, int rInt, int[] cx, int[] cy,
    int xStep, int yStep, double imgMax, double pruneBelow, boolean earlyExit)
  {
    if (xStep<1) xStep=1; if (yStep<1) yStep=1

    for (int k=0;k<N;k++){
      if (cx[k]<0||cx[k]>=W||cy[k]<0||cy[k]>=H) return Double.NEGATIVE_INFINITY
    }
    int dyLo=-rInt, dyHi=rInt
    for (int k=0;k<N;k++){
      int loK=-cy[k], hiK=(H-1)-cy[k]
      if (loK>dyLo) dyLo=loK; if (hiK<dyHi) dyHi=hiK
    }
    if (dyLo>dyHi) return Double.NEGATIVE_INFINITY

    int mod0=(dyLo + rInt) % yStep
    if (mod0!=0){ dyLo += (yStep - mod0); if (dyLo>dyHi) return Double.NEGATIVE_INFINITY }

    final int rowsTotal = Math.floorDiv(dyHi - dyLo, yStep) + 1
    final int approxSamplesPerRow = Math.max(1, Math.floorDiv((2*rInt + 1) + (xStep - 1), xStep))
    final double ubPerSample=imgMax

    double sum=0d; int[] rowBase=new int[N]; int rowsDone=0
    for (int dyOff=dyLo; dyOff<=dyHi; dyOff+=yStep){
      int half = hw[dyOff + rInt]
      int commonLeft  = -half
      int commonRight =  half
      int maxLeftBound = Integer.MIN_VALUE
      int minRightBound = Integer.MAX_VALUE
      for (int k=0;k<N;k++){
        int base=(cy[k]+dyOff)*W; rowBase[k]=base
        int leftBoundK  = -cx[k]
        int rightBoundK = (W-1) - cx[k]
        if (leftBoundK  > maxLeftBound)  maxLeftBound  = leftBoundK
        if (rightBoundK < minRightBound) minRightBound = rightBoundK
      }
      commonLeft  = Math.max(commonLeft,  maxLeftBound)
      commonRight = Math.min(commonRight, minRightBound)
      if (commonLeft <= commonRight){
        int oxStart=commonLeft
        int m=(oxStart+half) % xStep; if (m<0) m+=xStep; if (m!=0) oxStart += (xStep - m)
        for (int ox=oxStart; ox<=commonRight; ox+=xStep){
          double minv=Double.POSITIVE_INFINITY
          for (int k=0;k<N;k++){
            int idx=rowBase[k] + (cx[k]+ox)
            float v=p[idx]
            if (v<minv) minv=v
          }
          sum += minv
        }
      }
      rowsDone++
      if (earlyExit && pruneBelow!=Double.NEGATIVE_INFINITY){
        int rowsLeft=rowsTotal - rowsDone
        double ubRemain=rowsLeft * (double)approxSamplesPerRow * ubPerSample
        if (sum + ubRemain <= pruneBelow) return Double.NEGATIVE_INFINITY
      }
    }
    return sum
  }
}

// ---------------- General utilities (static) ----------------
@CompileStatic
class WellUtil {
  static double clamp(double v, double lo, double hi){ return Math.max(lo, Math.min(hi, v)) }

  static void overlayCirclesWithNumbers(ImagePlus imp, java.util.List<Roi> rois, Color circleColor, Color labelColor, Font font){
    Overlay ov = new Overlay()
    int idx = 1
    for (Roi r : rois){
      Roi rr = (Roi) r.clone()
      rr.setStrokeColor(circleColor); rr.setStrokeWidth(2.5f); rr.setFillColor(null)
      ov.add(rr)
      Rectangle b = r.getBounds()
      double cx = b.getX() + b.getWidth()/2.0d
      double cy = b.getY() + b.getHeight()/2.0d
      TextRoi tr
      try { tr = new TextRoi((int)Math.round(cx-6.0d), (int)Math.round(cy-6.0d), String.valueOf(idx)); tr.setCurrentFont(font) }
      catch(Throwable t){ tr = new TextRoi(cx-6.0d, cy-6.0d, String.valueOf(idx), font) }
      tr.setStrokeColor(labelColor); tr.setAntiAlias(true); ov.add(tr)
      idx++
    }
    imp.setOverlay(ov); imp.updateAndDraw()
  }

  static int[] labelMapFromRois(java.util.List<Roi> rois, int w, int h){
    int[] labels = new int[w*h]; java.util.Arrays.fill(labels, 0)
    for (int idx=0; idx<rois.size(); idx++){
      Roi r = rois.get(idx)
      Rectangle rb = r.getBounds()
      final int bx = (int)rb.getX()
      final int by = (int)rb.getY()
      final int bw = (int)rb.getWidth()
      final int bh = (int)rb.getHeight()
      ByteProcessor m = (ByteProcessor) r.getMask()
      if (m != null){
        byte[] mp = (byte[]) m.getPixels()
        for (int yy=0; yy<bh; yy++){
          final int rowOff = (by + yy) * w
          final int mOff = yy * bw
          for (int xx=0; xx<bw; xx++){
            if ((mp[mOff + xx] & 0xff) != 0){
              int i = rowOff + (bx + xx)
              if (i>=0 && i<labels.length && labels[i]==0) labels[i] = idx+1
            }
          }
        }
      } else {
        double rx = (double)bw/2.0d, ry = (double)bh/2.0d
        double cx = (double)bx + rx,   cy = (double)by + ry
        for (int yy=0; yy<bh; yy++){
          final int rowOff = (by + yy) * w
          double y = ((double)by + (double)yy + 0.5d - cy)/ry
          double y2 = y*y
          for (int xx=0; xx<bw; xx++){
            double x = ((double)bx + (double)xx + 0.5d - cx)/rx
            if (x*x + y2 <= 1.0d){
              int i = rowOff + (bx + xx)
              if (i>=0 && i<labels.length && labels[i]==0) labels[i] = idx+1
            }
          }
        }
      }
    }
    return labels
  }

  static int[] computeCropBounds(boolean[] inside, int w, int h, int pad){
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

  static java.util.List<Roi> shiftRois(java.util.List<Roi> rois, int dx, int dy){
    java.util.List<Roi> out = new java.util.ArrayList<>()
    for (Roi r : rois){
      Roi rr = (Roi) r.clone()
      Rectangle b = rr.getBounds()
      int bx = (int)Math.round(b.getX())
      int by = (int)Math.round(b.getY())
      rr.setLocation(bx - dx, by - dy)
      out.add(rr)
    }
    return out
  }

  static FloatProcessor toFloatFP(ImagePlus imp){
    ImagePlus work = imp.duplicate()
    if (work.getBitDepth()!=32) work.setProcessor(work.getProcessor().convertToFloatProcessor())
    return (FloatProcessor) work.getProcessor()
  }

  static int[] maskedHistogram(FloatProcessor fp, boolean[] inside, double lo, double hi, int bins){
    int n = fp.getWidth()*fp.getHeight()
    float[] p = (float[]) fp.getPixels()
    int[] h = new int[bins]
    double scale = (bins-1)/Math.max(1e-9d, (hi-lo))
    for (int i=0; i<n; i++){
      if (!inside[i]) continue
      double v = (double)p[i]
      int b = (int)Math.floor((v - lo)*scale + 0.5d)
      if (b<0) b=0; else if (b>=bins) b=bins-1
      h[b]++
    }
    return h
  }

  // --- Otsu threshold (no external class needed) ---
  static double otsuFromHist(int[] h, double lo, double hi){
    int bins = (h==null)?0:h.length
    if (bins < 2) return (lo+hi)*0.5d
    long total = 0L
    long sumAll = 0L
    for (int i=0;i<bins;i++){ int c=h[i]; total += (long)c; sumAll += (long)i * (long)c }
    if (total==0L) return (lo+hi)*0.5d
    long wB = 0L
    long sumB = 0L
    double maxVar = Double.NEGATIVE_INFINITY
    int bestT = 0
    for (int t=0; t<bins; t++){
      int c = h[t]
      wB += (long)c
      if (wB==0L) continue
      long wF = total - wB
      if (wF==0L) break
      sumB += (long)t * (long)c
      double mB = (double)sumB / (double) wB
      double mF = ((double)sumAll - (double)sumB) / (double) wF
      double diff = (mB - mF)
      double varBetween = (double)wB * (double)wF * diff * diff
      if (varBetween > maxVar){ maxVar = varBetween; bestT = t }
    }
    return lo + ((double)bestT / (double)(bins-1)) * (hi - lo)
  }

  static int[] buildBaseRGB(FloatProcessor fp, double lo, double hi){
    int w = fp.getWidth(), h = fp.getHeight(), n = w*h
    float[] src = (float[]) fp.getPixels()
    int[] base = new int[n]
    double scale = (hi > lo) ? (255.0d/(hi - lo)) : 1.0d
    for (int i=0; i<n; i++){
      int g = (int) Math.round(Math.max(0.0d, Math.min(255.0d, ((double)src[i]-lo)*scale )))
      base[i] = (0xFF<<24) | (g<<16) | (g<<8) | g
    }
    return base
  }

  static int[] buildTintedRGB(FloatProcessor fp, int[] baseRGB, boolean[] inside, double thr){
    int w = fp.getWidth(), h = fp.getHeight(), n = w*h
    float[] src = (float[]) fp.getPixels()
    int[] out = java.util.Arrays.copyOf(baseRGB, baseRGB.length)
    final int rT=255,gT=64,bT=160; final double alpha=0.60d, ia=1.0d-alpha
    final double dim=0.70d, brighten=1.30d
    for (int i=0; i<n; i++){
      if (!inside[i]) continue
      int px = out[i]; int g = px & 0xff
      if ((double)src[i] >= thr){
        int gDim = (int)Math.round((double)g * dim)
        int r2 = (int)Math.round(ia*gDim + alpha*rT)
        int g2 = (int)Math.round(ia*gDim + alpha*gT)
        int b2 = (int)Math.round(ia*gDim + alpha*bT)
        out[i] = (0xFF<<24) | ((r2&0xff)<<16) | ((g2&0xff)<<8) | (b2&0xff)
      } else {
        int gB = (int)Math.round(Math.min(255.0d, (double)g * brighten))
        out[i] = (0xFF<<24) | ((gB&0xff)<<16) | ((gB&0xff)<<8) | (gB&0xff)
      }
    }
    return out
  }

  static java.util.List<Roi> buildRigidGridRotated(double tlx,double tly,double dx,double dy,double diam,double angleDeg,int W,int H){
    int rInt = Math.max(2, (int)Math.round(diam/2.0d))
    double[] offX = new double[GridOpt.N], offY = new double[GridOpt.N]
    GridOpt.rotatedOffsets(dx, dy, angleDeg, offX, offY)
    double[] b = new double[4]
    GridOpt.feasibleAnchorBounds(W, H, offX, offY, rInt, b)
    double ax = clamp(tlx, b[0], b[1])
    double ay = clamp(tly, b[2], b[3])
    double r = (double) rInt
    java.util.List<Roi> rois = new java.util.ArrayList<>()
    for (int k=0; k<GridOpt.N; k++){
      double cx = ax + offX[k], cy = ay + offY[k]
      rois.add(new OvalRoi(cx - r, cy - r, 2.0d*r, 2.0d*r))
    }
    return rois
  }
}

// ---------------- Grid search (auto‑propose) ----------------
@CompileStatic
class GridSearch {
  @CompileStatic
  static class DS { FloatProcessor fp; double scale; DS(FloatProcessor fp,double scale){this.fp=fp;this.scale=scale;} }

  @CompileStatic
  static GridSearch.DS downscaleDS(FloatProcessor fp, int maxSide){
    int W=fp.getWidth(), H=fp.getHeight()
    double scale=1.0d; int longSide=Math.max(W,H)
    if (longSide>maxSide){
      scale = (double)maxSide / (double)longSide
      int newW=Math.max(1,(int)Math.round((double)W*scale))
      int newH=Math.max(1,(int)Math.round((double)H*scale))
      ImageProcessor rs
      try{ rs = fp.resize(newW,newH,true) }catch(Throwable t){ rs = fp.resize(newW,newH) }
      return new DS((FloatProcessor)rs.convertToFloatProcessor(), scale)
    } else return new DS(fp,1.0d)
  }

  @CompileStatic
  static class Best { double score; int ax; int ay; double dx; double dy; double ang; Best(double s){score=s;} }

  @CompileStatic
  static GridSearch.Best evaluateAnchorsMT(
    float[] p,int W,int H,
    int rInt,int[] hw,double[] offX,double[] offY,
    int ax0,int ax1,int ay0,int ay1,int aStep,
    int xStep,int yStep,double imgMax,boolean earlyExit,int nThreads)
  {
    final int N=GridOpt.N
    ExecutorService pool=Executors.newFixedThreadPool(Math.max(1,nThreads))
    java.util.List<Future<GridSearch.Best>> futs=new java.util.ArrayList<>()
    for (int t=0;t<nThreads;t++){
      final int tIdx=t
      futs.add(pool.submit(new Callable<GridSearch.Best>(){
        @Override GridSearch.Best call(){
          GridSearch.Best best=new GridSearch.Best(Double.NEGATIVE_INFINITY)
          int[] cx=new int[N], cy=new int[N]
          double bestSeen=Double.NEGATIVE_INFINITY
          for (int ay=ay0 + tIdx*aStep; ay<=ay1; ay+=aStep*nThreads){
            for (int ax=ax0; ax<=ax1; ax+=aStep){
              for (int k=0;k<N;k++){ cx[k]=(int)Math.round((double)ax+offX[k]); cy[k]=(int)Math.round((double)ay+offY[k]) }
              double sc=GridOpt.scoreMinProjSampledBound(p,W,H,hw,rInt,cx,cy,xStep,yStep,imgMax,bestSeen,earlyExit)
              if (sc>bestSeen){ bestSeen=sc; best.score=sc; best.ax=ax; best.ay=ay }
            }
          }
          return best
        }
      }))
    }
    GridSearch.Best globalBest=new GridSearch.Best(Double.NEGATIVE_INFINITY)
    for (Future<GridSearch.Best> f : futs){ try{ GridSearch.Best b=f.get(); if (b.score>globalBest.score) globalBest=b }catch(Throwable ignore){} }
    pool.shutdown(); return globalBest
  }

  @CompileStatic
  static double[] autoProposeMinProjLightning(ImagePlus imp,int defDiam,int startDx,int startDy,SpeedProfile prof){
    FloatProcessor fpFull = WellUtil.toFloatFP(imp)
    int W=fpFull.getWidth(), H=fpFull.getHeight()
    float[] pFull=(float[])fpFull.getPixels()
    double imgMaxFull=fpFull.getStatistics().max

    GridSearch.DS ds=downscaleDS(fpFull, prof.MAX_SIDE)
    FloatProcessor fpS=ds.fp; double s=ds.scale
    int Ws=fpS.getWidth(), Hs=fpS.getHeight()
    float[] pS=(float[])fpS.getPixels()
    double imgMaxS=fpS.getStatistics().max

    double diam=(double)defDiam, diamS=diam*s
    int rS=Math.max(2,(int)Math.round(diamS/2.0d))
    int[] hwS=GridOpt.buildHalfWidths(rS)

    int rF=Math.max(2,(int)Math.round((double)defDiam/2.0d))
    int[] hwF=GridOpt.buildHalfWidths(rF)

    double[] muls=new double[]{0.85d,0.95d,1.05d,1.15d}
    double angMin=-prof.ANG_SPAN, angMax=prof.ANG_SPAN, angStep=prof.ANG_STEP0
    int aStep=Math.max(10,(int)Math.round((diamS/2.0d)*prof.COARSE_STRIDE_MUL))

    GridSearch.Best bestC=new GridSearch.Best(Double.NEGATIVE_INFINITY)
    int nThreads=Math.max(1,Runtime.getRuntime().availableProcessors())

    double[] offX=new double[GridOpt.N], offY=new double[GridOpt.N], bounds=new double[4]

    // coarse (downscaled)
    for (int mi=0; mi<muls.length; mi++){
      double dxS=Math.max(1.0d, diamS*muls[mi])
      double dyS=dxS
      for (double ang=angMin; ang<=angMax; ang+=angStep){
        GridOpt.rotatedOffsets(dxS,dyS,ang,offX,offY)
        GridOpt.feasibleAnchorBounds(Ws,Hs,offX,offY,rS,bounds)
        if (bounds[0]>bounds[1] || bounds[2]>bounds[3]) continue
        int ax0=(int)Math.round(bounds[0]), ax1=(int)Math.round(bounds[1])
        int ay0=(int)Math.round(bounds[2]), ay1=(int)Math.round(bounds[3])
        GridSearch.Best b=evaluateAnchorsMT(pS,Ws,Hs,rS,hwS,offX,offY,ax0,ax1,ay0,ay1,aStep,
                                 prof.XSTEP_C,prof.YSTEP_C,imgMaxS,prof.EARLY_EXIT,nThreads)
        if (b.score>bestC.score){ bestC=b; bestC.dx=dxS; bestC.dy=dyS; bestC.ang=ang }
      }
    }

    // fine (downscaled neighborhood)
    GridSearch.Best bestF1=new GridSearch.Best(Double.NEGATIVE_INFINITY)
    int aStepFine=Math.max(4,(int)Math.round((double)aStep/3.0d))
    double base=(bestC.dx+bestC.dy)*0.5d
    double[] dCand=new double[7]; for(int i=0;i<7;i++) dCand[i]=Math.max(1.0d, base*(0.88d+0.04d*i))
    for(double ang=bestC.ang-3.0d; ang<=bestC.ang+3.0d; ang+=0.25d){
      for(int i=0;i<7;i++) for(int j=0;j<7;j++){
        double dxS=dCand[i], dyS=dCand[j]
        GridOpt.rotatedOffsets(dxS,dyS,ang,offX,offY)
        GridOpt.feasibleAnchorBounds(Ws,Hs,offX,offY,rS,bounds)
        if (bounds[0]>bounds[1] || bounds[2]>bounds[3]) continue
        int ax0=(int)Math.round(bounds[0]), ax1=(int)Math.round(bounds[1])
        int ay0=(int)Math.round(bounds[2]), ay1=(int)Math.round(bounds[3])
        int axSeed=(int)Math.round(WellUtil.clamp((double)bestC.ax,bounds[0],bounds[1]))
        int aySeed=(int)Math.round(WellUtil.clamp((double)bestC.ay,bounds[2],bounds[3]))
        int axA=Math.max(ax0, axSeed-3*aStepFine), axB=Math.min(ax1, axSeed+3*aStepFine)
        int ayA=Math.max(ay0, aySeed-3*aStepFine), ayB=Math.min(ay1, aySeed+3*aStepFine)
        GridSearch.Best b=evaluateAnchorsMT(pS,Ws,Hs,rS,hwS,offX,offY,axA,axB,ayA,ayB,aStepFine,
                                 prof.XSTEP_F,prof.YSTEP_F,imgMaxS,prof.EARLY_EXIT,nThreads)
        if (b.score>bestF1.score){ bestF1=b; bestF1.dx=dxS; bestF1.dy=dyS; bestF1.ang=ang }
      }
    }

    // tiny full‑res neighborhood
    double tlxSeed=(double)bestF1.ax/s, tlySeed=(double)bestF1.ay/s, dxSeed=bestF1.dx/s, dySeed=bestF1.dy/s, angSeed=bestF1.ang
    GridSearch.Best bestFinal=new GridSearch.Best(Double.NEGATIVE_INFINITY)
    int stepPos=Math.max(3,(int)Math.round((double)defDiam/18.0d)), stepSp=2
    for(double ang=angSeed-1.0d; ang<=angSeed+1.0d; ang+=0.25d){
      for(int dyi=-2; dyi<=2; dyi++){
        double dyTry=Math.max(1.0d, dySeed + (double)dyi*stepSp)
        for(int dxi=-2; dxi<=2; dxi++){
          double dxTry=Math.max(1.0d, dxSeed + (double)dxi*stepSp)
          double[] offX2=new double[GridOpt.N], offY2=new double[GridOpt.N]
          GridOpt.rotatedOffsets(dxTry,dyTry,ang,offX2,offY2)
          double[] bnds=new double[4]
          GridOpt.feasibleAnchorBounds(W,H,offX2,offY2,rF,bnds)
          if (bnds[0]>bnds[1] || bnds[2]>bnds[3]) continue
          int ax0=(int)Math.round(WellUtil.clamp(tlxSeed,bnds[0],bnds[1]))
          int ay0=(int)Math.round(WellUtil.clamp(tlySeed,bnds[2],bnds[3]))
          int axA=Math.max((int)Math.round(bnds[0]), ax0-2*stepPos)
          int axB=Math.min((int)Math.round(bnds[1]), ax0+2*stepPos)
          int ayA=Math.max((int)Math.round(bnds[2]), ay0-2*stepPos)
          int ayB=Math.min((int)Math.round(bnds[3]), ay0+2*stepPos)
          int[] cx=new int[GridOpt.N], cy=new int[GridOpt.N]
          for (int ay=ayA; ay<=ayB; ay+=stepPos){
            for (int ax=axA; ax<=axB; ax+=stepPos){
              for (int k=0;k<GridOpt.N;k++){ cx[k]=(int)Math.round((double)ax+offX2[k]); cy[k]=(int)Math.round((double)ay+offY2[k]) }
              double sc=GridOpt.scoreMinProjSampledBound(pFull,W,H,hwF,rF,cx,cy,
                                                         prof.FINAL_XSTEP,prof.FINAL_YSTEP,
                                                         imgMaxFull,bestFinal.score,prof.EARLY_EXIT)
              if (sc>bestFinal.score){
                bestFinal.score=sc; bestFinal.ax=ax; bestFinal.ay=ay
                bestFinal.dx=dxTry; bestFinal.dy=dyTry; bestFinal.ang=ang
              }
            }
          }
        }
      }
    }
    return [ (double)bestFinal.ax, (double)bestFinal.ay, bestFinal.dx, bestFinal.dy, bestFinal.ang ] as double[]
  }
}

// ---------------- Histogram panel (Stage‑2) ----------------
class HistPanel extends JPanel {
  int[] hist; double thr; double lo; double hi
  HistPanel(int[] h, double t, double lo, double hi){ this.hist=h; this.thr=t; this.lo=lo; this.hi=hi; setPreferredSize(new Dimension(320,120)) }
  void setThr(double t){ thr=t; repaint() }
  @Override protected void paintComponent(Graphics g){
    super.paintComponent(g)
    Graphics2D g2 = (Graphics2D) g
    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    int W = getWidth(), H = getHeight()
    g2.setColor(new Color(245,245,245)); g2.fillRect(0,0,W,H)
    g2.setColor(new Color(200,200,200)); g2.drawRect(0,0,W-1,H-1)
    if (hist==null || hist.length==0) return
    int bins = hist.length, maxC = 1; for (int v: hist) if (v>maxC) maxC=v
    int x0=32, y0=H-20, ww=W-42, hh=H-30
    g2.setColor(new Color(90,90,90))
    int px=-1, py=-1
    for (int i=0; i<bins; i++){
      double xf = (double)i/(double)(bins-1)
      int x = x0 + (int)Math.round(xf*(double)(ww-1))
      int y = y0 - (int)Math.round((double)hh * ((double)hist[i]/(double)maxC))
      if (px>=0) g2.drawLine(px,py,x,y)
      px=x; py=y
    }
    double tf = (thr - lo) / Math.max(1e-9d, (hi - lo)); tf = Math.max(0.0d, Math.min(1.0d, tf))
    int xT = x0 + (int)Math.round(tf*(double)(ww-1))
    g2.setColor(Color.RED); g2.drawLine(xT, y0-hh, xT, y0)
  }
}

// ---------------- UI controller ----------------
class RigidGrid3x2UI {
  // Class‑local defaults (no dependency on script scope)
  static final int    DEFAULT_DIAM   = 240
  static final int    FALLBACK_TLX   = 254
  static final int    FALLBACK_TLY   = 248
  static final int    FALLBACK_DX    = 261
  static final int    FALLBACK_DY    = 261
  static final double FALLBACK_ANG   = 0d
  static final SpeedProfile SP       = new SpeedProfile(360, 5,5, 3,3, 2,2, 3.0d, 8.0d, 1.00d, true)

  ImagePlus imp; int W,H
  int diamVar; double angVar

  // UI
  JDialog dlg
  JSpinner spTLX, spTLY, spDX, spDY
  JRadioButton rbN10, rbN1, rbDia10, rbDia1, rbAng1, rbAng01
  JLabel lblDia, lblAng
  JButton btnDm, btnDp, btnAm, btnAp, btnAuto, btnOK, btnCancel
  JButton btnLeft, btnRight, btnUp, btnDown, btnDXm, btnDXp, btnDYm, btnDYp

  // Steps
  int nudge = 10
  int diaStep = 10
  double angStep = 1.0d

  RigidGrid3x2UI(ImagePlus imp){ this.imp=imp; this.W=imp.getWidth(); this.H=imp.getHeight() }

  void refreshOverlay(){
    int tlx=(int)spTLX.getValue(), tly=(int)spTLY.getValue(), dx=(int)spDX.getValue(), dy=(int)spDY.getValue()
    java.util.List<Roi> rois = WellUtil.buildRigidGridRotated((double)tlx,(double)tly,(double)dx,(double)dy,(double)diamVar,angVar,W,H)
    WellUtil.overlayCirclesWithNumbers(imp, rois, Color.red, Color.white, new Font("SansSerif", Font.BOLD, 13))
    lblDia.setText("Diameter: "+diamVar+" px")
    lblAng.setText(String.format("θ = %.2f°", angVar))
  }

  void onAutoPropose(){
    btnAuto.setEnabled(false)
    new Thread(new Runnable(){
      public void run(){
        try{
          IJ.showStatus("Rigid 3×2: auto‑proposing (lightning)…")
          double[] best = GridSearch.autoProposeMinProjLightning(imp, diamVar, FALLBACK_DX, FALLBACK_DY, SP)
          spTLX.setValue((int)Math.round(best[0]))
          spTLY.setValue((int)Math.round(best[1]))
          spDX.setValue((int)Math.round(best[2]))
          spDY.setValue((int)Math.round(best[3]))
          angVar = best[4]
          SwingUtilities.invokeLater(new Runnable(){ public void run(){ refreshOverlay() }})
          IJ.showStatus("Rigid 3×2: proposal applied.")
        } catch(Throwable t){ IJ.handleException(t) }
        finally { SwingUtilities.invokeLater(new Runnable(){ public void run(){ btnAuto.setEnabled(true) }}) }
      }
    }, "AutoProposeLightning").start()
  }

  void openStage2AndAnalyze(){
    int tlx=(int)spTLX.getValue(), tly=(int)spTLY.getValue(), dx=(int)spDX.getValue(), dy=(int)spDY.getValue()
    java.util.List<Roi> rois = WellUtil.buildRigidGridRotated((double)tlx,(double)tly,(double)dx,(double)dy,(double)diamVar,angVar,W,H)

    ImagePlus workImp = imp.duplicate()
    if (workImp.getBitDepth()!=32) workImp.setProcessor(workImp.getProcessor().convertToFloatProcessor())
    int W0=workImp.getWidth(), H0=workImp.getHeight()
    final int[] labelMap = WellUtil.labelMapFromRois(rois, W0, H0)
    boolean[] insideMask = new boolean[W0*H0]
    for (int i=0; i<insideMask.length; i++) insideMask[i] = (labelMap[i] != 0)

    int[] cb = WellUtil.computeCropBounds(insideMask, W0, H0, 8)
    int x0=cb[0], y0=cb[1], CW=cb[2], CH=cb[3]

    float[] srcAll = (float[])((FloatProcessor)workImp.getProcessor()).getPixels()
    final float[] dstCrop = new float[CW*CH]
    final boolean[] insideCrop = new boolean[CW*CH]
    for (int y=0; y<CH; y++){
      int sy = y0 + y
      for (int x=0; x<CW; x++){
        int sx = x0 + x
        int si = sy*W0 + sx
        int dii = y*CW + x
        dstCrop[dii] = insideMask[si] ? srcAll[si] : 0f
        insideCrop[dii] = insideMask[si]
      }
    }
    final FloatProcessor wellsFP = new FloatProcessor(CW, CH, dstCrop)
    final ImagePlus wellsImp = new ImagePlus(imp.getTitle()+" [wells-only,cropped]", wellsFP)
    java.util.List<Roi> shifted = WellUtil.shiftRois(rois, x0, y0)
    WellUtil.overlayCirclesWithNumbers(wellsImp, shifted, Color.red, Color.white, new Font("SansSerif", Font.BOLD, 13))
    try { IJ.run(wellsImp, "Original Scale", "") } catch(Throwable ignore) {}
    wellsImp.show()

    // Stage‑2 UI
    def st = wellsFP.getStatistics()
    final double lo = (Double.isFinite(st.min) && Double.isFinite(st.max) && st.min!=st.max) ? (double)st.min : 0.0d
    final double hi = (Double.isFinite(st.min) && Double.isFinite(st.max) && st.min!=st.max) ? (double)st.max : 255.0d
    int[] h = WellUtil.maskedHistogram(wellsFP, insideCrop, lo, hi, 256)
    final double thr0 = WellUtil.otsuFromHist(h, lo, hi)
    final int[] baseRGB = WellUtil.buildBaseRGB(wellsFP, lo, hi)
    final HistPanel histPanel = new HistPanel(h, thr0, lo, hi)

    final JDialog d2 = new JDialog((Frame)null, "Stage‑2: Contents – threshold & preview", true)
    d2.setLayout(new BorderLayout(10,10))
    JPanel t2 = new JPanel(); t2.setLayout(new BoxLayout(t2, BoxLayout.Y_AXIS))
    JLabel title2 = new JLabel("Set contents threshold; vivid tint shows segmented area")
    title2.setFont(new Font("SansSerif", Font.BOLD, 16)); title2.setAlignmentX(JLabel.CENTER_ALIGNMENT)
    t2.add(title2); t2.add(Box.createVerticalStrut(6))

    final JSlider thrSlider = new JSlider(0, 1000, (int)Math.round(1000.0d*((thr0-lo)/Math.max(1e-9d, hi-lo))))
    final JLabel thrLabel = new JLabel(String.format("Content threshold: %.3f", thr0), JLabel.CENTER)
    JPanel thrPanel = new JPanel(new BorderLayout(6,6))
    thrPanel.add(thrLabel, BorderLayout.NORTH); thrPanel.add(thrSlider, BorderLayout.CENTER)
    t2.add(thrPanel)

    JPanel statsPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 12, 4))
    statsPanel.add(histPanel)
    t2.add(Box.createVerticalStrut(6)); t2.add(statsPanel)

    d2.add(t2, BorderLayout.CENTER)
    JPanel btn2 = new JPanel(new FlowLayout(FlowLayout.RIGHT))
    JButton cancel2 = new JButton("Cancel"); JButton ok2 = new JButton("OK")
    btn2.add(cancel2); btn2.add(ok2); d2.add(btn2, BorderLayout.SOUTH)
    d2.pack(); d2.setLocationRelativeTo(null)

    final Runnable applyTint = new Runnable(){
      @Override public void run(){
        double thr = lo + ((double)thrSlider.getValue()/1000.0d)*(hi-lo)
        thrLabel.setText(String.format("Content threshold: %.3f", thr))
        int[] tinted = WellUtil.buildTintedRGB(wellsFP, baseRGB, insideCrop, thr)
        histPanel.setThr(thr)
        ColorProcessor cp = new ColorProcessor(CW, CH, tinted)
        ImageRoi imgLayer = new ImageRoi(0,0, cp); imgLayer.setOpacity(1.0f)
        Overlay ov = new Overlay(); ov.add(imgLayer)
        int idx=1
        for (Roi r : shifted){
          Roi rr = (Roi) r.clone()
          rr.setStrokeColor(Color.red); rr.setStrokeWidth(2.5f); rr.setFillColor(null); ov.add(rr)
          Rectangle b = rr.getBounds()
          double cx = b.getX() + b.getWidth()/2.0d, cy = b.getY() + b.getHeight()/2.0d
          TextRoi tr = new TextRoi((int)Math.round(cx-6.0d), (int)Math.round(cy-6.0d), String.valueOf(idx))
          tr.setCurrentFont(new Font("SansSerif", Font.BOLD, 13)); tr.setStrokeColor(Color.white); tr.setAntiAlias(true); ov.add(tr)
          idx++
        }
        wellsImp.setOverlay(ov); wellsImp.updateAndDraw()
      }
    }

    applyTint.run()
    thrSlider.addChangeListener(new ChangeListener(){ public void stateChanged(ChangeEvent e){ applyTint.run() }})

    ok2.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e2){
        double thr = lo + ((double)thrSlider.getValue()/1000.0d)*(hi-lo)
        // Mean intensity of segmented pixels per well on the FULL image
        FloatProcessor srcFull = (FloatProcessor) imp.getProcessor().convertToFloatProcessor()
        float[] pFull = (float[]) srcFull.getPixels()
        long[] countSel = new long[7]
        double[] sumSel = new double[7]
        for (int y=0; y<H; y++){
          int off = y*W
          for (int x=0; x<W; x++){
            int idx2 = off + x
            int label = labelMap[idx2]
            if (label==0) continue
            double v = (double)pFull[idx2]
            if (v>=thr){ countSel[label]++; sumSel[label]+=v }
          }
        }
        ResultsTable rt = new ResultsTable()
        for (int widx=1; widx<=6; widx++){
          double mean = (countSel[widx]==0L) ? 0.0d : (sumSel[widx]/(double)countSel[widx])
          rt.incrementCounter()
          rt.addValue("Well #", widx)
          rt.addValue("Pixels (seg)", (double)countSel[widx])
          rt.addValue("Mean Intensity (seg)", Math.round(mean*1000.0d)/1000.0d)
        }
        rt.show("Well Analysis — Mean Intensity")
        d2.dispose()
      }
    })
    cancel2.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e2){ d2.dispose() }})
    d2.setVisible(true)
  }

  void run(){
    // Initial auto‑propose (no UI/overlay yet)
    IJ.showStatus("Rigid 3×2: auto‑proposing grid (lightning)…")
    double[] bestInit
    try {
      bestInit = GridSearch.autoProposeMinProjLightning(imp, DEFAULT_DIAM, FALLBACK_DX, FALLBACK_DY, SP)
    } catch(Throwable t){
      IJ.handleException(t)
      bestInit = [ (double)FALLBACK_TLX, (double)FALLBACK_TLY, (double)FALLBACK_DX, (double)FALLBACK_DY, FALLBACK_ANG ] as double[]
    }
    IJ.showStatus("Rigid 3×2: proposal ready.")

    diamVar = DEFAULT_DIAM
    angVar  = bestInit[4]

    // ---- Build Stage‑1 UI ----
    dlg = new JDialog((Frame)null, "Rigid 3×2 Grid (Lightning)", true)
    dlg.setLayout(new BorderLayout(10,10))
    JPanel main = new JPanel(); main.setLayout(new BoxLayout(main, BoxLayout.Y_AXIS))
    JLabel hdr = new JLabel("Adjust the 3×2 grid; diameter & angle via buttons below")
    hdr.setFont(new Font("SansSerif", Font.BOLD, 16)); hdr.setAlignmentX(Component.CENTER_ALIGNMENT)
    main.add(hdr); main.add(Box.createVerticalStrut(6))

    // Hidden spinners for position/spacing (prefilled from proposal)
    spTLX = new JSpinner(new SpinnerNumberModel((int)Math.round(bestInit[0]), 0, Math.max(1,W), 1))
    spTLY = new JSpinner(new SpinnerNumberModel((int)Math.round(bestInit[1]), 0, Math.max(1,H), 1))
    spDX  = new JSpinner(new SpinnerNumberModel((int)Math.round(bestInit[2]),  1, Math.max(1,W), 1))
    spDY  = new JSpinner(new SpinnerNumberModel((int)Math.round(bestInit[3]),  1, Math.max(1,H), 1))

    // Step selectors row
    JPanel rowSteps = new JPanel(new FlowLayout(FlowLayout.LEFT, 16, 2))
    // Nudge
    JPanel pNudge = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0))
    pNudge.add(new JLabel("Nudge:"))
    rbN10 = new JRadioButton("10 px", true); rbN1 = new JRadioButton("1 px")
    ButtonGroup bgN = new ButtonGroup(); bgN.add(rbN10); bgN.add(rbN1)
    pNudge.add(rbN10); pNudge.add(rbN1); rowSteps.add(pNudge)
    // Diameter step
    JPanel pDiaStep = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0))
    pDiaStep.add(new JLabel("Diameter step:"))
    rbDia10 = new JRadioButton("10", true); rbDia1 = new JRadioButton("1")
    ButtonGroup bgDia = new ButtonGroup(); bgDia.add(rbDia10); bgDia.add(rbDia1)
    pDiaStep.add(rbDia10); pDiaStep.add(rbDia1); rowSteps.add(pDiaStep)
    // Angle step
    JPanel pAngStep = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0))
    pAngStep.add(new JLabel("Angle step:"))
    rbAng1 = new JRadioButton("1°", true); rbAng01 = new JRadioButton("0.1°")
    ButtonGroup bgAng = new ButtonGroup(); bgAng.add(rbAng1); bgAng.add(rbAng01)
    pAngStep.add(rbAng1); pAngStep.add(rbAng01); rowSteps.add(pAngStep)
    main.add(rowSteps)

    // Diameter row
    JPanel rowDia = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 2))
    lblDia = new JLabel("Diameter: "+diamVar+" px")
    btnDm = new JButton("D −"); btnDp = new JButton("D +")
    rowDia.add(lblDia); rowDia.add(btnDm); rowDia.add(btnDp)
    main.add(rowDia)

    // Angle row
    JPanel rowAng = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 2))
    lblAng = new JLabel(String.format("θ = %.2f°", angVar))
    btnAm = new JButton("Ang −"); btnAp = new JButton("Ang +")
    rowAng.add(lblAng); rowAng.add(btnAm); rowAng.add(btnAp)
    main.add(rowAng)

    // Position arrows
    JPanel nudgePos = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2))
    nudgePos.add(new JLabel("Position:"))
    btnLeft = new JButton("←"); btnRight = new JButton("→"); btnUp = new JButton("↑"); btnDown = new JButton("↓")
    nudgePos.add(btnLeft); nudgePos.add(btnRight); nudgePos.add(btnUp); nudgePos.add(btnDown)
    main.add(nudgePos)

    // Spacing
    JPanel nudgeSp = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2))
    nudgeSp.add(new JLabel("Spacing:"))
    btnDXm = new JButton("dx −"); btnDXp = new JButton("dx +"); btnDYm = new JButton("dy −"); btnDYp = new JButton("dy +")
    nudgeSp.add(btnDXm); nudgeSp.add(btnDXp); nudgeSp.add(btnDYm); nudgeSp.add(btnDYp)
    main.add(nudgeSp)

    // Buttons
    JPanel buttons = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 2))
    btnAuto = new JButton("Auto Propose")
    btnOK   = new JButton("OK")
    btnCancel = new JButton("Cancel")
    buttons.add(btnAuto); buttons.add(btnOK); buttons.add(btnCancel)
    main.add(buttons)

    dlg.add(main, BorderLayout.CENTER)
    dlg.pack(); dlg.setLocationRelativeTo(null)

    // Listeners
    rbN10.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ nudge=10 }})
    rbN1 .addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ nudge=1  }})
    rbDia10.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ diaStep=10 }})
    rbDia1 .addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ diaStep=1  }})
    rbAng1 .addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ angStep=1.0d  }})
    rbAng01.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ angStep=0.1d }})

    btnLeft .addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ spTLX.setValue(Math.max(0, ((int)spTLX.getValue()) - nudge)); refreshOverlay() }})
    btnRight.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ spTLX.setValue(((int)spTLX.getValue()) + nudge); refreshOverlay() }})
    btnUp   .addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ spTLY.setValue(Math.max(0, ((int)spTLY.getValue()) - nudge)); refreshOverlay() }})
    btnDown .addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ spTLY.setValue(((int)spTLY.getValue()) + nudge); refreshOverlay() }})

    btnDXm.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ int v=(int)spDX.getValue(); v=Math.max(1, v - nudge); spDX.setValue(v); refreshOverlay() }})
    btnDXp.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ int v=(int)spDX.getValue(); v=v + nudge; spDX.setValue(v); refreshOverlay() }})
    btnDYm.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ int v=(int)spDY.getValue(); v=Math.max(1, v - nudge); spDY.setValue(v); refreshOverlay() }})
    btnDYp.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ int v=(int)spDY.getValue(); v=v + nudge; spDY.setValue(v); refreshOverlay() }})

    btnDm.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ diamVar = Math.max(4, diamVar - diaStep); refreshOverlay() }})
    btnDp.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ diamVar = Math.min(Math.min(W,H), diamVar + diaStep); refreshOverlay() }})

    btnAm.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ angVar = angVar - angStep; refreshOverlay() }})
    btnAp.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ angVar = angVar + angStep; refreshOverlay() }})

    btnAuto.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ onAutoPropose() }})
    btnCancel.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ imp.setOverlay(null); dlg.dispose() }})
    btnOK.addActionListener(new ActionListener(){ public void actionPerformed(ActionEvent e){ dlg.dispose(); openStage2AndAnalyze() }})

    // Show overlay + dialog (after proposal)
    refreshOverlay()
    dlg.setVisible(true)
  }
}

// ---------------- Main ----------------
ImagePlus imp = IJ.getImage()
if (imp==null){ IJ.error("Open an image first."); return }
new RigidGrid3x2UI(imp).run()
