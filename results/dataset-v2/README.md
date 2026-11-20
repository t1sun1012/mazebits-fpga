# Structured maze dataset

This dataset extends training coverage while preserving the 3x8x8 binary input and four-action BFS target contract. Generate it with `python python/prepare_structured_data.py`; `data/maze-v2` must be new or empty. The [manifest](manifest.json) records generation settings, file hashes, canonical layout identities, exclusions, and family/distance audits.

| Split | Layouts | States | Rollout episodes | Maximum BFS distance |
|---|---:|---:|---:|---:|
| Train | 1,200 | 163,128 | 28,800 | 36 |
| Validation | 200 | 27,084 | 4,800 | 34 |
| Reserved test | 200 | 27,612 | 4,800 | 35 |

Training has 480 independent random-obstacle layouts, 240 connected barriers/elbows/U-shaped layouts, and 480 winding corridors with alternating gaps. The corridor generator requires sampled endpoint separation of at least 26 moves. There are 4,897 training states with shortest-path distance at least 26.

Each layout has three goals: two far-apart BFS-sweep endpoints and a random third goal. The two sweeps do not claim an exact graph diameter. Every reachable nonterminal state receives all optimal BFS actions. Eight distance-stratified rollout starts per goal include an actual farthest start. This changes the goal/start distribution as well as obstacle shapes compared with v1.

All 1,100 v1 wall layouts and six distinct known demonstration/photograph layouts are excluded, including their rotations/reflections. Every new layout stays entirely within its split. The known photographs remain development diagnostics, not an unbiased test estimate. Training/validation tools never open the reserved-test arrays or episodes; only their generation audit is recorded here.
