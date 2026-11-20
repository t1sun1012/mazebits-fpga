# Structured-model navigation reference vectors

Seven scenarios exercise both physically distinct precision variants of the Step 7 checkpoint. Every action score was checked against the independent scalar integer reference before export. These vectors provide expected outputs to the RTL testbench; they do not assist the board policy.

| Scenario | W8A8 outcome / attempts | W4A4 outcome / attempts |
|---|---|---|
| open | success / 14 | success / 14 |
| detour | success / 21 | loop / 8 |
| blocked | loop / 3 | loop / 5 |
| at-goal | success / 0 | success / 0 |
| precision-reversal-IMG_7179 | success / 16 | success / 16 |
| long-detour-IMG_7180 | success / 26 | loop / 4 |
| precision-loss-latest-photo | success / 14 | success / 16 |

The bundle contains 7 scenarios, 14 episodes, and 157 independently checked decisions. All four signed scores, actions, positions, trails, statuses, attempted-action counts, and cumulative inference cycles are recorded. Each completed decision is expected to take 3,217,153 clock cycles. Zero-step start/goal overlap performs no inference.

```powershell
python python/export_structured_navigation_vectors.py --output-dir runs/reexport-structured-navigation
python scripts/run_navigation.py --model-set structured --sim xsim --vivado-bin C:/Xilinx/Vivado/2024.1/bin
```

Export requires a new or empty output directory. The runner verifies model identities, source hashes, complete case/episode coverage, and exact vector file sizes and word counts before simulation. The original four-case model/vector suite remains available with `--model-set original`.

The photographed maze cases are known development diagnostics; they are not an unbiased final-test estimate. No reserved dataset partition is read.
