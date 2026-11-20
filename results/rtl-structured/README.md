# Step 8 structured-model core verification

[Verification JSON](verification.json) is copied byte-for-byte from the final local XSim run. [W8A8 simulation log](w8a8_simulation.txt) and [W4A4 simulation log](w4a4_simulation.txt) use LF line endings.

Both improved models passed Vivado XSim 2024.1 with the unchanged authored CNN RTL. Each precision checks eight fixtures and 133,160 intermediate/output words, plus one repeated inference (16,645 more words). Each also passes 13 rounding-boundary cases, three signed/tie argmax cases, and all seven recorded protocol checks.

Both require 3,217,153 cycles per inference: 32.17153 ms at a nominal 100 MHz. The complete runner took 31.167 s for W8A8 and 30.061 s for W4A4 on this workstation; these are verification wall times, not FPGA inference times. Core simulation does not establish routed timing or physical-board behavior.

The nine focused model-selection/provenance tests passed, including mutation after manifest validation and during staged-memory writing. The runner validates and stages captured source bytes, records hashes of the intended materialized bytes, and checks source and staged hashes before reporting success.

Final `scripts/run_rtl.py` SHA-256:

```text
7508880e3a3655c76f583c8e109424395a5afd9a79aef85eaab1684e58cc9362
```

Reproduce from the repository root:

```powershell
python -m unittest discover -s python/tests -p test_rtl_model_selection.py -v
python scripts/run_rtl.py --model-set structured --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin --bits 8 4
```

These results use `model/cnn-structured-w8a8` and `model/cnn-structured-w4a4` with their matching frozen reference vectors. The original-model evidence remains in `results/rtl`.
