# Microscope Tile Stitching — Assignment Neuranics
*(By Priyanshu Srivastava)*  

---

## Overview

This is a small research project I worked on to stitch about **70 microscope image tiles** into one large, high-resolution composite.  
Each image was captured at the same focal depth but with a small overlap (roughly 8%).  

I wanted to build everything from scratch using **OpenCV** instead of relying on **Ashlar** or any commercial stitching tool.  
The main idea was to use **feature matching (SIFT + RANSAC)** for geometric alignment and then blend the images smoothly with **Feather blending**.

---

## Goals

- Build a working microscopy tile stitching pipeline **entirely in OpenCV**.  
- Use **RANSAC-based homography estimation** to align overlapping tiles.  
- Avoid massive memory consumption that happens with OpenCV’s `Stitcher` or multiband blending.  
- Produce a clean final image (and an OME-TIFF) that looks as close to a continuous tissue slide as possible.

---

## Technical Summary

### Grid Setup
- Total tiles: **70**
- Arrangement: **10 rows × 7 columns**
- Image size: **1920 × 1200 px**
- Tile order: **serpentine scan** (left to right, then right to left)
- Overlap: about 8%

### Alignment Process
Each image aligns with its **left** and **bottom** neighbors.

I used:
- **SIFT feature detection** (or ORB if SIFT isn’t available)
- **Brute-force matcher** + **RANSAC** to estimate homography
- Fallback to **phase correlation** or **ECC affine** alignment if too few features were detected

This combination helped in recovering sub-pixel translations between tiles even when some regions had low contrast or tissue uniformity.

### Refinement
After all local alignments, I averaged offsets with a small weighting logic to avoid sudden jumps between tiles.  
This keeps the grid structure intact while still allowing local flexibility.

### Blending
At first, I tried **MultiBandBlender** and **GraphCutSeamFinder**, but they immediately ran out of memory (over 100 GB!).  
So I switched to **FeatherBlender**, which is much lighter.  
It needs the images as `int16`, and after blending, I converted the result back to `uint8`.

To fix black seams and small missing patches, I used a combination of:
- Mean color filling  
- Morphological cleanup  
- `cv2.inpaint()` for the final touch

---

## Environment & Tools

- Python 3.13  
- OpenCV 4.12.0  
- tifffile 2024.x  
- ome-types 0.4+  
- Tested on 16 GB RAM system (Windows)

---

## Problems I Ran Into

| Problem | What Happened | What I Did |
|----------|----------------|------------|
| **Memory explosion (500+ GB)** | Multiband blending + seam finder were too heavy for 12k×11k canvas | Switched to `FeatherBlender` |
| **Few feature matches** | Plain tissue regions had little texture | Increased overlap area and lowered ratio threshold |
| **Tiles drifting apart** | Small cumulative alignment errors | Added weighted row/column correction |
| **Black gaps in output** | Some missing pixels after blending | Mean-color filling + inpainting |
| **FeatherBlender assertion error** | Expected 16-bit signed input | Converted tiles using `.astype(np.int16)` |

---

## Learnings & Takeaways

- OpenCV’s built-in `Stitcher` isn’t reliable for microscope data — overlaps are too small.  
- Doing feature matching manually gives more control and better results.  
- RANSAC homography helps filter bad matches, but homogeneous regions are still tricky.  
- Feather blending is a good middle ground between quality and performance.  
- Even after successful blending, **exposure mismatches** between microscope captures can cause visible seams — future work could normalize that.  

---

## Current Results

- **Final size:** 12,516 × 11,136 px  
- **Tiles stitched:** 70  
- **Output:** `stitched_registered.jpg` and `.ome.tif`  
- **Visual quality:** Mostly seamless, minor exposure differences remain  
- **Memory usage:** Manageable (< 8 GB peak)

---

## Future Work

- Global optimization (bundle adjustment) across all tiles  
- GPU-accelerated SIFT for faster keypoint extraction  
- Adaptive exposure matching across tiles  
- Export to DeepZoom or Zarr format for easy panning/zooming  
- Optionally integrate microscope stage metadata to verify alignment

---

## Files

| File | Description |
|------|--------------|
| `opencv_Stitch.py` | Full OpenCV stitching code |
| `stitched_registered.jpg` | Final composite image |
| `stitched_registered.ome.tif` | Metadata-rich OME-TIFF |
| `Microscope_Stitching_RnD_Report.pdf` | Printable version of this documentation |
| `README_RnD.md` | This document |

---

## Final Thoughts

This was a good experiment assignment in understanding how commercial stitching tools like Ashlar actually work under the hood.  
It’s still not perfect — some edges are off, and there are light seams here and there — but it’s impressive how far OpenCV can go with the right combination of RANSAC, ECC, and blending.
  
For now, this pipeline gives me a transparent and reproducible way to stitch microscope slides entirely offline — no external dependencies, no hidden black boxes. This is the best i thought of time is running out i tried ashlar but i wasnt able to achieve, so i looked uo the internet youtube and ai, and found my way through above mentioned methodolgy.

---

*Written and developed by*  
**Priyanshu Srivastava**  
2025
