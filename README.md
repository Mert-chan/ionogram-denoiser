# Denoising Low-Latitude Ionograms Without Losing Spread-F Echoes

This is a heuristic denoiser designed to remove noise from the ionogram images of the SEALION network of NICT. Spread-F cases were especially considered during the design process. The variety of cases where the denoiser works, and its failure modes, are discussed in the arXiv paper.


![Steps of the denoiser on a spread-F frame (CPN 2025-08-23 15:05 UTC).](figures/fig1_method.png)


`denoiser.py` => the denoiser. `load_crop(png)` reads a raw ionogram, `denoise(crop)` returns the echo mask (True = echo). All thresholds are at the top of the file.
On average, the denoiser takes 36 ms per frame (Intel Core i7-6800K).

Data availability: SEALION ionograms are available from NICT at https://aer-nc-web.nict.go.jp/sealion/ subject to the SEALION data policy.

Acknowledgment: The CPN and BCL ionosondes are operated by NICT, Japan, within the SEALION network, in collaboration with KMITL, Thailand, and IES-VAST, Vietnam.

