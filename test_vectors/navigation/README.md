# Closed-loop navigation vectors

These four scenarios are evaluated by both frozen integer CNNs. Every expected score is independently checked by the scalar fixed-point oracle during export. The paired RTL test elaborates real W8A8/W4A4 engines together and compares every completed decision.

`manifest.json` records scenario definitions, full episode events, source/model hashes, every memory file's size/hash, and the packing contract. `walls.mem`, `starts.mem`, and `goals.mem` provide four inputs in the order open, detour, disconnected, and start-at-goal. Cell index is row*8+column.

Each `events_w*a*.mem` contains 64 event slots per scenario and 11 32-bit words per slot: action, resulting position, trail low/high, status, attempted actions, accumulated inference cycles, then UP/RIGHT/DOWN/LEFT signed scores as two's-complement words. Unused event slots are zero-filled. Each `final_w*a*.mem` contains six words per scenario: status, attempts, position, trail low/high, and total inference cycles.

Status codes are 0 READY, 1 RUN, 2 GOAL, 3 BLOCKED, 4 LOOP, 5 LIMIT. An illegal action leaves the position unchanged and still counts as an attempt. Goal arrival takes priority over the limit; a repeated legal position terminates a deterministic stateless policy. Zero-distance scenarios call no CNN.

Regenerate with the project ML environment from the checkout root:

~~~powershell
python python/export_navigation_vectors.py
~~~

The original trained models and held-out dataset are unchanged. Run `python scripts/run_navigation.py --sim iverilog --tests cnn` to compare the generated dual engines with these vectors without installing Python ML libraries.
