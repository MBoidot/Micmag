# Nd_spacing — README

Overview
--------
This subfolder contains the Python port of the original MATLAB image analysis pipeline for measuring local Nd-rich spacing and orientations from micrographs. The main entrypoint is the ANALYSE_IMAGES.py script that:
- reads images,
- performs contrast enhancement & background-corrected binarisation,
- computes spacing (local mean distances) and orientations,
- computes distributions/histograms and exports results per image.

Pipeline (high level)
---------------------
1. Image & parameters:
   - Main script: [ANALYSE_IMAGES.py](Nd_spacing/ANALYSE_IMAGES.py)
   - Global settings: [PARAMETRES.py](Nd_spacing/PARAMETRES.py)

2. Binarisation:
   - CLAHE + background correction + thresholding (Otsu or adaptive).
   - Implementation: [`binarisation`](Nd_spacing/Binarisation_methods/binarisation%20clahe+BGcorr+adaptive%20thresholding.py)

3. Spacing & statistics:
   - Local distance map: [`distance`](Nd_spacing/distance.py)
   - Normalized histogram / distribution: [`distribution`](Nd_spacing/distribution.py)

4. Orientation:
   - Orientation/angle calculation: [`calculangle`](Nd_spacing/calculangle.py)

5. Post-processing / export:
   - Visualization and XLS export: [`dessinexport`](Nd_spacing/dessinexport.py)
   - Grain prolongation/cleanup helpers: [`prolong_nd`](Nd_spacing/prolong_nd.py)

Files of interest
-----------------
- [ANALYSE_IMAGES.py](Nd_spacing/ANALYSE_IMAGES.py) — main orchestration script, loops samples/types/photos and calls the pipeline steps.
- [PARAMETRES.py](Nd_spacing/PARAMETRES.py) — dataset/sample parameters (ECHTS, TYPES, NBPHOTOS, etc.).
- [Nd_spacing/Binarisation_methods/binarisation clahe+BGcorr+adaptive thresholding.py](Nd_spacing/Binarisation_methods/binarisation%20clahe+BGcorr+adaptive%20thresholding.py) — detailed binarisation routine.
- [distance.py](Nd_spacing/distance.py) — compute local mean spacing map.
- [distribution.py](Nd_spacing/distribution.py) — convert DM map into histogram classes and normalized distribution.
- [calculangle.py](Nd_spacing/calculangle.py) — compute local crystal/grain orientations and class histograms.
- [dessinexport.py](Nd_spacing/dessinexport.py) — saves figures and XLSX results.
- [prolong_nd.py](Nd_spacing/prolong_nd.py) — morphological helpers used for grain prolongation and cleaning.

Quick start
-----------
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Configure [PARAMETRES.py](Nd_spacing/PARAMETRES.py) for your dataset (sample names, photo counts, paths).
3. Run the analyzer from the Nd_spacing folder:
   ```sh
   python ANAYLSE_IMAGES.py
   ```
   (Script prints progress and writes per-image XLSX in the "Résultats" folder.)

Configuration notes
-------------------
- Binarisation parameters (CLAHE clip_limit, tile_grid_size, blur kernel, threshold method, adaptive block_size/C) are configurable in the binarisation call made by [ANALYSE_IMAGES.py](Nd_spacing/ANALYSE_IMAGES.py).
- Distribution parameters such as `dmax` and `pasdistri` are imported from [PARAMETRES.py](Nd_spacing/PARAMETRES.py) and used by [`distribution`](Nd_spacing/distribution.py).

Implementation quirks & pointers
-------------------------------
- The pipeline mirrors the original MATLAB flow (see Code_matlab_OT_original for the MATLAB reference).
- The binarisation routine applies CLAHE then optional Gaussian background estimation before thresholding.
- [`distribution`](Nd_spacing/distribution.py) computes histogram class boundaries consistent with the MATLAB original.
- If you want to inspect intermediate outputs, [ANALYSE_IMAGES.py](Nd_spacing/ANALYSE_IMAGES.py) writes masks and labeled images (TIFF) for debugging.

# external folder

[text](https://github.com/zsylvester/segmenteverygrain)