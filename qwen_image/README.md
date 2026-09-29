# Qwen-Image backend

This directory contains the FreqCa integration for Qwen-Image and
Qwen-Image-Edit. See the [repository README](../README.md) for installation,
single-GPU, multi-GPU, and GEdit-Bench commands.

The pipeline files are derived from Hugging Face Diffusers and keep their
upstream Apache-2.0 headers. FreqCa is injected through `cache_functions/` and
the `cache_dic` / `current` arguments in the custom pipelines.
