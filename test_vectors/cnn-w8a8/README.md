# W8A8 RTL reference vectors

Eight cases are stored in the order recorded by manifest.json: open corners, adjacent goal, long detour, disconnected map, retained validation loop, initial goal, two free cells, and the first training episode. No final-test examples are used.

Each file contains one hex word per line. Cases are concatenated first, then channel/row/column (CHW). Weight layout is separately OIHW in the model bundle.

| File group | Words | Width and meaning |
|---|---:|---|
| inputs.mem | 1,536 | 1-bit binary inputs |
| Each convN_acc.mem | 8,192 | Signed 32-bit accumulator |
| Each convN_act.mem | 8,192 | Unsigned 8-bit activation |
| head_acc.mem | 2,048 | Signed 32-bit full action field |
| scores.mem | 32 | Four signed 32-bit scores per case |
| actions.mem | 8 | Unsigned 2-bit lowest-index argmax |

All 133,160 intermediate/output words matched independent Python-integer loops, which check every partial sum. Written memory words were then reloaded and verified. The manifest binds every file and the source model with SHA-256.

The initial-goal case checks arithmetic; a navigation controller should terminate when already at the goal.

~~~powershell
python python/verify_fixed_vectors.py --model-dir model/cnn-w8a8 --vectors-dir test_vectors/cnn-w8a8
~~~

Verification needs only the committed model and vectors. Dataset generation is needed to regenerate the training fixture, not to replay the frozen vectors. These are reference values for Step 4; no RTL simulation is claimed here.
