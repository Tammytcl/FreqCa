# FLUX backend

This directory contains the FreqCa integration for FLUX.1-dev,
FLUX.1-schnell, FLUX.1-Fill-dev, and FLUX.1-Kontext-dev. See the
[repository README](../README.md) for installation, inference, distributed
sampling, and evaluation commands.

The `flux` Python package is derived from the official Black Forest Labs FLUX
inference repository. FreqCa's implementation lives under
`src/flux/modules/cache_functions/` and is called from `src/flux/sampling.py`.
