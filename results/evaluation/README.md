# Improved-model quantization evaluation

The [PDF report](../../output/pdf/MazeBits_Quantization_Evaluation.pdf) and [Markdown version](../../docs/quantization-evaluation.md) compare the same improved checkpoint at W8A8 and W4A4. Their evidence snapshot is commit `0bd19baa7dea709d8711b16bfb94b5be9aa4c523` from Step 8.

The report adds paired success counts and maze-family/BFS-distance analyses from existing saved validation trajectories. It performs no retraining, new model inference, or reserved-test evaluation. Quality metrics are software reference results; latency is derived from verified cycles and nominal 100 MHz timing closure; FPGA resources are routed measurements; board checks are user-reported physical observations.

- [metrics.json](metrics.json): source hashes, exact paired counts, subgroup denominators, hardware metrics, and physical observations.
- [Comparison figure](comparison.png) and [vector figure](comparison.svg): four panels with explicit units and zero-based axes.
- [Render record](report-render.json): PDF/Markdown hashes, page count, generator identity, and text/visual review status.

## Reproduce

Use the project Python environment for evidence aggregation and matplotlib:

```powershell
python scripts/prepare_evaluation_report.py
```

The dataset's hash-verified `data/maze-v2/val.json` supplies explicit maze-family metadata. If the dataset is absent, regenerate it with `python python/prepare_structured_data.py --output-dir data/maze-v2`; reuse an existing dataset rather than overwriting it. The report script only opens validation metadata, not the reserved test partition.

Render with a Python environment containing `reportlab` and `pypdf`:

```powershell
python scripts/render_evaluation_report.py
```

On the task computer, the project environment supplies matplotlib and the bundled document runtime supplies ReportLab. Render the resulting PDF using Poppler and inspect all five pages before publication. The rendering script also produces the Markdown source and rejects changed evidence or chart hashes. Its layout is intended for this frozen evaluation, not arbitrary replacement data.

On the 4,040 routes solved by both models, W8A8 uses 33,566 moves and W4A4 uses 33,652. Their mean nominal inference-only route times are 267.29445 ms and 267.97929 ms. This paired comparison avoids treating a model's shorter failed routes or a different successful subset as a speed advantage.
