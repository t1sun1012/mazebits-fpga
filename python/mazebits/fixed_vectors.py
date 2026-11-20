"""Complete integer-CNN trace fixtures for independent verification and RTL."""

import json
from math import prod
from pathlib import Path
import re

import numpy as np

from .dataset import walls_from_bits
from .encoding import encode_state
from .examples import EXAMPLES
from .fixed_point import infer, input_codes
from .maze import Maze
from .model_io import sha256, write_json
from .quantized_io import load_quantized
from .scalar_fixed import infer_scalar

_FORMAT = "mazebits-fixed-vectors-v1"
_CASE_NAMES = (
    "open", "adjacent-goal", "detour", "blocked", "validation-loop",
    "at-goal", "two-free-cells", "training-first",
)
_LAYOUT = "case-major; NCHW tensors, case/action scores, then one action per case"


def _integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _cases(training_metadata_path: Path | None) -> tuple[list[dict], np.ndarray, dict]:
    root = Path(__file__).resolve().parents[2]
    training_path = (root / "data/maze-v1/train.json"
                     if training_metadata_path is None else Path(training_metadata_path))
    metadata = json.loads(training_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("split") != "train":
        raise ValueError("trace selection may read only training metadata")
    episodes, layouts = metadata.get("episodes"), metadata.get("layouts")
    if not isinstance(episodes, list) or not episodes or not isinstance(layouts, list):
        raise ValueError("training metadata needs layouts and at least one episode")
    episode = episodes[0]
    layout_index = _integer(episode.get("layout_index"), "layout_index")
    if layout_index >= len(layouts):
        raise ValueError("training episode has an invalid layout_index")
    layout = layouts[layout_index]
    goal_index = _integer(episode.get("goal_index"), "goal_index")
    goals = layout.get("goals")
    if (not isinstance(goals, list) or goal_index >= len(goals)
            or episode.get("goal") != goals[goal_index]):
        raise ValueError("training episode goal disagrees with its layout")
    training_maze = Maze(walls_from_bits(layout["walls"]),
                         tuple(episode["start"]), tuple(episode["goal"]))
    loop_path = root / "examples/validation-loop.txt"
    free = {(0, 0), (1, 0)}
    dense_walls = frozenset((x, y) for y in range(8) for x in range(8)
                            if (x, y) not in free)
    mazes = (
        EXAMPLES["open"],
        Maze(frozenset(), (3, 3), (4, 3)),
        EXAMPLES["detour"],
        EXAMPLES["blocked"],
        Maze.from_ascii(loop_path.read_text(encoding="utf-8")),
        EXAMPLES["at-goal"],
        Maze(dense_walls, (0, 0), (1, 0)),
        training_maze,
    )
    cases = [{"name": name, "maze_ascii": maze.to_ascii()}
             for name, maze in zip(_CASE_NAMES, mazes)]
    inputs = np.asarray([encode_state(maze, maze.start) for maze in mazes],
                        dtype=np.int64).reshape(len(mazes), 3, 8, 8)
    sources = {
        "training_metadata_sha256": sha256(training_path),
        "training_episode_index": 0,
        "validation_loop_fixture_sha256": sha256(loop_path),
        "selection_note": "one retained validation diagnostic; no test examples",
    }
    return cases, inputs, sources


def _specs(policy, count: int) -> dict[str, dict]:
    specs = {
        "inputs": {"shape": [count, 3, 8, 8], "bits": 1, "signed": False},
    }
    for layer in policy.layers:
        shape = [count, int(layer.weight.shape[0]), 8, 8]
        specs[f"{layer.name}.acc"] = {"shape": shape, "bits": 32, "signed": True}
        if layer.output_frac is not None:
            specs[f"{layer.name}.act"] = {
                "shape": shape, "bits": policy.bits, "signed": False,
            }
    specs["scores"] = {"shape": [count, 4], "bits": 32, "signed": True}
    specs["actions"] = {"shape": [count], "bits": 2, "signed": False}
    return specs


def _filename(key: str) -> str:
    return key.replace(".", "_") + ".mem"


def _write_words(path: Path, values: np.ndarray, bits: int, signed: bool) -> None:
    values = np.asarray(values)
    lower = -(1 << (bits - 1)) if signed else 0
    upper = (1 << (bits - int(signed))) - 1
    if values.dtype.kind not in "iu" or (values < lower).any() or (values > upper).any():
        raise ValueError(f"trace values exceed declared integer width: {path.name}")
    digits, mask = (bits + 3) // 4, (1 << bits) - 1
    path.write_text("".join(f"{int(value) & mask:0{digits}x}\n" for value in values.flat),
                    encoding="ascii", newline="\n")


def _read_words(path: Path, spec: dict) -> np.ndarray:
    try:
        rows = path.read_bytes().decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"non-ASCII trace file: {path.name}") from error
    if len(rows) != prod(spec["shape"]):
        raise ValueError(f"incorrect trace word count: {path.name}")
    bits = spec["bits"]
    pattern = re.compile(rf"[0-9a-fA-F]{{{(bits + 3) // 4}}}")
    values = []
    for index, row in enumerate(rows):
        if pattern.fullmatch(row) is None:
            raise ValueError(f"invalid hex word in {path.name} at {index}")
        value = int(row, 16)
        if value >= 1 << bits:
            raise ValueError(f"word exceeds its declared width in {path.name}")
        if spec["signed"] and value & (1 << (bits - 1)):
            value -= 1 << bits
        values.append(value)
    return np.asarray(values, dtype=np.int64).reshape(spec["shape"])


def _model_identity(model_dir: Path) -> dict:
    return {
        "manifest_sha256": sha256(model_dir / "manifest.json"),
        "weights_sha256": sha256(model_dir / "weights.mem"),
        "biases_sha256": sha256(model_dir / "biases.mem"),
        "header_sha256": sha256(model_dir / "quantization.vh"),
    }


def _compare(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.shape != expected.shape:
        raise ValueError(f"trace shape mismatch: {label}")
    count = int(np.count_nonzero(actual != expected))
    if count:
        raise ValueError(f"trace mismatch: {label} ({count} words)")


def _scalar_checks(policy, inputs, traces) -> int:
    words = 0
    for index in range(len(inputs)):
        scores, scalar = infer_scalar(policy, inputs[index:index + 1])
        _compare(scores, traces["scores"][index:index + 1], f"scalar scores case {index}")
        if scalar.keys() != traces.keys():
            raise ValueError("scalar/vector trace names do not match")
        for key in traces:
            _compare(scalar[key], traces[key][index:index + 1], f"{key} scalar case {index}")
            words += int(scalar[key].size)
    return words


def export_vectors(
    model_dir: Path, output_dir: Path, training_metadata_path: Path | None = None
) -> dict:
    """Reload memory weights, independently check eight cases, then export traces."""
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"vector output must be new or empty: {output_dir}")
    policy = load_quantized(model_dir)
    cases, inputs, sources = _cases(training_metadata_path)
    _, traces = infer(policy, inputs, trace=True)
    compared_words = _scalar_checks(policy, inputs, traces)
    values = {"inputs": inputs, **traces}
    specs = _specs(policy, len(cases))
    if values.keys() != specs.keys():
        raise ValueError("inference trace keys do not match the export contract")
    manifest = {
        "format": _FORMAT, "bits": policy.bits, "case_count": len(cases),
        "layout": _LAYOUT, "cases": cases, "fixture_sources": sources,
        "source_model": _model_identity(model_dir),
        "files": {},
        "verification": {"cases": len(cases), "compared_words": compared_words,
                         "input_words": int(inputs.size), "mismatches": 0,
                         "method": "all intermediate words matched independent scalar loops"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, spec in specs.items():
        filename = _filename(key)
        if list(values[key].shape) != spec["shape"]:
            raise ValueError(f"trace shape disagrees with export contract: {key}")
        _write_words(output_dir / filename, values[key], spec["bits"], spec["signed"])
        manifest["files"][filename] = {
            "trace": key, **spec, "words": prod(spec["shape"]),
            "words_per_case": prod(spec["shape"][1:]),
            "sha256": sha256(output_dir / filename),
            "bytes": (output_dir / filename).stat().st_size,
        }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def verify_vectors(
    model_dir: Path, vectors_dir: Path, case_limit: int | None = None
) -> dict:
    """Verify all files and compare selected cases with vector and scalar engines."""
    model_dir, vectors_dir = Path(model_dir), Path(vectors_dir)
    policy = load_quantized(model_dir)
    manifest = json.loads((vectors_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT:
        raise ValueError("unsupported fixed-vector format")
    if type(manifest.get("bits")) is not int or manifest["bits"] != policy.bits:
        raise ValueError("vector precision does not match the model")
    count = _integer(manifest.get("case_count"), "case_count", 1)
    if count != len(_CASE_NAMES) or manifest.get("layout") != _LAYOUT:
        raise ValueError("invalid fixture count or tensor layout")
    if case_limit is not None:
        _integer(case_limit, "case_limit", 1)
        if case_limit > count:
            raise ValueError("case_limit exceeds the available fixtures")
    selected = count if case_limit is None else case_limit
    sources = manifest.get("fixture_sources")
    if (not isinstance(sources, dict)
            or any(not isinstance(sources.get(key), str)
                   or re.fullmatch(r"[0-9a-f]{64}", sources[key]) is None
                   for key in ("training_metadata_sha256", "validation_loop_fixture_sha256"))
            or _integer(sources.get("training_episode_index"), "training_episode_index") != 0):
        raise ValueError("invalid fixture source metadata")
    if manifest.get("source_model") != _model_identity(model_dir):
        raise ValueError("vectors were exported for a different model bundle")
    cases = manifest.get("cases")
    if (not isinstance(cases, list) or len(cases) != count
            or any(not isinstance(case, dict) for case in cases)
            or tuple(case.get("name") for case in cases) != _CASE_NAMES):
        raise ValueError("fixture names/count/order are inconsistent")
    expected_inputs = []
    for case in cases:
        text = case.get("maze_ascii")
        if not isinstance(text, str):
            raise ValueError("fixture needs an ASCII maze")
        maze = Maze.from_ascii(text)
        expected_inputs.append(encode_state(maze, maze.start))
    expected_inputs = np.asarray(expected_inputs, dtype=np.int64).reshape(count, 3, 8, 8)
    specs = _specs(policy, count)
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {_filename(key) for key in specs}:
        raise ValueError("trace file set does not match the model")
    decoded = {}
    for key, spec in specs.items():
        filename = _filename(key)
        record = files[filename]
        if not isinstance(record, dict):
            raise ValueError(f"invalid trace metadata: {filename}")
        shape = record.get("shape")
        if (not isinstance(shape, list)
                or any(type(value) is not int or value < 1 for value in shape)
                or shape != spec["shape"]):
            raise ValueError(f"invalid trace shape: {filename}")
        if (record.get("trace") != key
                or type(record.get("bits")) is not int
                or record["bits"] != spec["bits"]
                or record.get("signed") is not spec["signed"]):
            raise ValueError(f"invalid trace layout/width/signedness: {filename}")
        if (_integer(record.get("words"), "trace words", 1) != prod(shape)
                or _integer(record.get("words_per_case"), "case words", 1) != prod(shape[1:])):
            raise ValueError(f"inconsistent trace word metadata: {filename}")
        path = vectors_dir / filename
        if _integer(record.get("bytes"), "trace bytes", 1) != path.stat().st_size:
            raise ValueError(f"trace file length mismatch: {filename}")
        if sha256(path) != record.get("sha256"):
            raise ValueError(f"trace file hash mismatch: {filename}")
        decoded[key] = _read_words(path, spec)
    input_codes(decoded["inputs"])  # Validate every stored state, even for a limited replay.
    _compare(decoded["inputs"], expected_inputs, "inputs versus fixture metadata")
    selected_inputs = decoded["inputs"][:selected]
    _, traces = infer(policy, selected_inputs, trace=True)
    expected_trace_keys = set(decoded) - {"inputs"}
    if set(traces) != expected_trace_keys:
        raise ValueError("vector trace names do not match the memory contract")
    for key in traces:
        _compare(traces[key], decoded[key][:selected], f"{key} memory versus vector")
    compared_words = _scalar_checks(policy, selected_inputs, traces)
    total_words = sum(prod(spec["shape"]) for key, spec in specs.items() if key != "inputs")
    recorded = manifest.get("verification")
    if (not isinstance(recorded, dict)
            or _integer(recorded.get("cases"), "verified cases", 1) != count
            or _integer(recorded.get("compared_words"), "compared words", 1) != total_words
            or _integer(recorded.get("input_words"), "input words", 1) != int(decoded["inputs"].size)
            or _integer(recorded.get("mismatches"), "mismatches") != 0):
        raise ValueError("export verification metadata is inconsistent")
    return {
        "cases": selected, "total_cases": count,
        "compared_words": compared_words, "input_words": int(selected_inputs.size),
        "mismatches": 0,
        "all_file_hashes_verified": True,
        "source_model": manifest["source_model"],
        "vectors_manifest_sha256": sha256(vectors_dir / "manifest.json"),
        "method": "memory words versus vector engine and independent scalar loops",
    }
