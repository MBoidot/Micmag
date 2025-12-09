# Orientation_methods — README

Purpose
-------
Contains alternative implementations to compute local orientations from a binary Nd-rich mask. Each script implements a strategy to produce:
- Per-pixel dominant angle map (degrees),
- Classes: vertical coordinate array (rows),
- distri: per-row summary (average deviation from vertical).

Files
-----
- calculangle_vectorized.py  
  - Vectorized, per-pixel scorer: precomputes offsets for candidate angles and tests all angles per pixel with array operations. Outputs DM, Classes, distri. Designed to replace the original calculangle.py for better performance without changing the analyzer's expected interface.
- calculangle_radon.py
  - Implements the radon algorithm for orientation calculation. Returns the global orientation of the image
- calculangle_translated_from_matlab.py
  - direct transcription of O.TOSONI Script
- calculangle_vector_full.py
  - Use of the structure tensor for orientation computation. No loops, no per pixel operation. Use of cv2.Sobel

Notes & requirements
--------------------
- PARAMETRES.py must expose at least:
  - pas (angle step, degrees)
  - envergure (sampling half-span, pixels)
- The methods assume a binary image (0/1). Ensure binarisation produces 1 for features and 0 for background.
- Sampling windows are 3 pixels wide across the tested line; adjust code if features are thicker.

How to make the analyzer use one of these methods
------------------------------------------------
Replace the content of the calculangle.py in the main folder by one of the scripts