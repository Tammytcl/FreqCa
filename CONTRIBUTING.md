# Contributing

Thanks for helping improve FreqCa. Please open an issue before a large change
so that implementation and evaluation scope can be agreed on early.

For a pull request:

1. Keep model-specific code inside `flux/` or `qwen_image/`.
2. Do not commit checkpoints, generated images, datasets, credentials, or local
   absolute paths.
3. Run `python -m compileall -q flux qwen_image scripts` and
   `python scripts/check_release.py`.
4. For algorithm changes, report the model, sampling steps, cache interval,
   decomposition method, resolution, GPU, latency measurement protocol, and at
   least one quality metric.
