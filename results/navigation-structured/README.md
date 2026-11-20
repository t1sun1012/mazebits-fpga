# Improved-weight FPGA verification and implementation

Step 8 integrates the Step 7 structured-maze checkpoint into the existing dual-CNN navigation hardware. The arithmetic core, controls, and VGA design remain unchanged. Original model bundles and results remain separate.

## Verification

- [Software tests](python-tests.txt): 153 pass; [source hashes and run record](python-test-verification.json).
- [Individual cores](../rtl-structured/README.md): eight fixtures and 133,160 intermediate/output words per precision, repeat and protocol checks, plus rounding and argmax boundary cases.
- [Navigation simulation](verification.json): seven scenarios, 14 episodes, 157 decisions, 628 exact signed-score matches; 202 controller checks, 34 top-level checks, and three complete VGA frames.
- [Original checkpoint regression](original-regression/verification.json): four scenarios, eight episodes, 60 decisions, and 240 score matches.

[Simulation evidence](simulation-evidence.json) preserves records from source commit `c4968ff0316dca7a4f0b64a68abb8d67e3f4bbcb`. The later staging integrity guard was checked by [fresh-stage equivalence](final-stager-equivalence.json): the six actual generated HDL/memory files for each model set are byte-identical to the simulated inputs. Historical verification hashes were not relabeled. The final implementation consumes those same six improved inputs.

## Implemented hardware

The final [Vivado implementation](build/implementation.json) passes the 100 MHz constraint: setup +0.603 ns, hold +0.042 ns, and zero DRC violations. Whole-design resources are 2,176 LUTs, 1,695 registers, 14.5 BRAM36 equivalents, and zero DSP blocks. The [hierarchy](build/utilization_hierarchy.rpt) assigns 632 LUTs / 9 BRAM equivalents to the 8-bit CNN and 524 LUTs / 5 BRAM equivalents to the 4-bit CNN. Both take 3,217,153 cycles per decision.

The [benchmark record](benchmark.json) joins these hardware measurements to the same exported models' structured-validation results: 4,766/4,800 successful routes at 8 bits and 4,050/4,800 at 4 bits. It keeps software quality measurements distinct from physical observations. See the [benchmarking guide](../../docs/benchmarking.md) for interpretation and the original checkpoint comparison.

The separate bitstream is `runs/vivado/navigation-structured/mazebits_navigation_structured.bit`, SHA-256 `3aa08afa503ab2ac9856d48f5c5aa702bbe53fa5b03d158081285be3a606536b`. The [GUI verification](build/gui_verification.json) confirms the separate project opens/reopens with the correct model memories and sources. Original bitstream/project hashes are unchanged. Bitstreams, generated GUI projects, and native simulation images remain local and can be recreated using [Step 8](../../docs/step-08.md).

## Physical observations and provenance

The user reports two improved-build board checks passed: both open-maze panels show GOAL / TRIES 014 / CYC HEX 02AF420E; the detour shows 8-bit GOAL / 021 / 0406E315 and 4-bit LOOP / 008 / 0188B808. Further tests are tracked separately in [board-test.json](board-test.json). Expected reference outcomes in that file are not physical observations until their status and observed fields are filled.

Build/GUI records are snapshots taken before programming and retain their build-time pending flags. The board-test log is the authoritative later physical record. Published Vivado reports omit local host/path details; [publication.json](build/publication.json) records raw and published hashes separately. Simulation verification JSON is preserved unchanged, and text logs use LF line endings.
