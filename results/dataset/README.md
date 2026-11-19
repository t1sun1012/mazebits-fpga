# Generated maze dataset

The committed manifest records the seed, generation rules, array/metadata hashes, split sizes, and layout identities for the Step 2 dataset. Generate the local arrays with:

~~~powershell
python python/prepare_data.py --output-dir data/maze-v1
~~~

The output directory must be new or empty. The generated manifest should match this file in the recorded environment.

| Split | Layouts | Goals | Labeled states | Fixed episodes |
|---|---:|---:|---:|---:|
| Training | 800 | 2,400 | 117,171 | 19,200 |
| Validation | 150 | 450 | 22,029 | 3,600 |
| Reserved test | 150 | 450 | 22,308 | 3,600 |

There are 1,100 unique wall layouts after grouping rotations and reflections. Goals and all associated states stay within their layout split. The generator rejected 31 candidates with insufficient connected free space.

Wall probabilities are sampled from 10%, 20%, and 30%. Each retained layout has a free component of at least 24 cells. Three distinct goals are selected in its largest component. BFS supplies all equally optimal actions for every reachable nonterminal state.

The final-test files were generated and hashed, but their arrays and episodes were not loaded for model selection or inference. The full arrays remain in the ignored data/ directory; this small manifest is sufficient to audit a regenerated dataset.

See [the Step 2 walkthrough](../../docs/step-02.md) for the encoding and training details.
