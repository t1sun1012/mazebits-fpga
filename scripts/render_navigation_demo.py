"""Render verified detour reference final states through the VGA RTL.

This is a VGA rendering of verified reference final states, not a full-system
CNN capture or a board photograph. The generated testbench contains no CNN.
Only Python's standard library and a Verilog simulator are required.

python scripts/render_navigation_demo.py --vivado-bin C:/Xilinx/Vivado/2024.1/bin
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import zlib
from run_navigation import executable, run, sha, verify_vector_bundle

# Keep X: mapped drive paths: Xilinx .bat tools cannot use a UNC working directory.
ROOT = Path(__file__).absolute().parents[1]
OUTCOME = {"success": 2, "blocked": 3, "loop": 4, "timeout": 5}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)


def convert_frame(source: Path, target: Path) -> dict:
    """Lossless P3 RGB444-expanded frame to RGB888 PNG, without image libraries."""
    words = source.read_text(encoding="ascii").split()
    if words[:4] != ["P3", "640", "480", "255"] or len(words) != 4 + 640 * 480 * 3:
        raise ValueError("incomplete or malformed VGA frame")
    values = [int(word) for word in words[4:]]
    if any(value < 0 or value > 255 or value % 17 for value in values):
        raise ValueError("frame contains invalid expanded RGB444 pixels")
    pixels = bytes(values)
    scanlines = b"".join(b"\0" + pixels[y*1920:(y+1)*1920] for y in range(480))
    payload = zlib.compress(scanlines, 9)
    ihdr = struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0)
    encoded = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) +
               png_chunk(b"IDAT", payload) + png_chunk(b"IEND", b""))
    target.write_bytes(encoded)
    # Independently parse the written PNG chunks, check CRCs, and round-trip RGB.
    written = target.read_bytes()
    if written[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("PNG signature changed")
    offset, compressed = 8, bytearray()
    while offset < len(written):
        length = struct.unpack(">I", written[offset:offset+4])[0]
        kind = written[offset+4:offset+8]
        data = written[offset+8:offset+8+length]
        crc = struct.unpack(">I", written[offset+8+length:offset+12+length])[0]
        if (zlib.crc32(kind + data) & 0xffffffff) != crc:
            raise AssertionError("PNG chunk checksum failed")
        if kind == b"IDAT": compressed.extend(data)
        offset += 12 + length
    decoded = zlib.decompress(compressed)
    if len(decoded) != 480 * 1921 or any(decoded[y*1921] != 0 for y in range(480)):
        raise AssertionError("PNG row encoding failed")
    recovered = b"".join(decoded[y*1921+1:(y+1)*1921] for y in range(480))
    if recovered != pixels:
        raise AssertionError("PNG pixels differ from simulated VGA frame")
    return {"width": 640, "height": 480, "pixel_count": 307200,
            "rgb_sha256": hashlib.sha256(pixels).hexdigest(), "png_pixel_identical": True}


def make_testbench(scenario: dict, episodes: dict) -> str:
    declarations = []
    for bits in (8, 4):
        episode = episodes[bits]
        cycles = episode["events"][-1]["total_cycles"] if episode["events"] else 0
        declarations.extend([
            f"    wire [5:0] agent{bits} = 6'd{episode['final_position']};",
            f"    wire [63:0] trail{bits} = 64'h{episode['trail_hex']};",
            f"    wire [2:0] status{bits} = 3'd{OUTCOME[episode['outcome']]};",
            f"    wire [6:0] steps{bits} = 7'd{episode['steps']};",
            f"    wire [31:0] cycles{bits} = 32'd{cycles};",
        ])
    return """`timescale 1ns/1ps
// Generated reference-state rendering: no inference core is instantiated.
module tb_navigation_preview;
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1;
""" + "\n".join(declarations) + f"""
    wire [3:0] red,green,blue;
    wire hsync,vsync,pixel_tick,active,frame_start;
    wire [9:0] pixel_x,pixel_y;
    maze_navigation_vga dut (
        .clk(clk),.rst(rst),.walls(64'h{scenario['walls_hex']}),
        .start_cell(6'd{scenario['start']}),.goal_cell(6'd{scenario['goal']}),
        .cursor(6'd{scenario['start']}),.mode(2'd3),.rejected(1'b0),.run_valid(1'b1),
        .agent8(agent8),.agent4(agent4),.trail8(trail8),.trail4(trail4),
        .status8(status8),.status4(status4),.steps8(steps8),.steps4(steps4),
        .cycles8(cycles8),.cycles4(cycles4),.red(red),.green(green),.blue(blue),
        .hsync(hsync),.vsync(vsync),.pixel_tick(pixel_tick),
        .pixel_x(pixel_x),.pixel_y(pixel_y),.active(active),.frame_start(frame_start)
    );
    integer frame=-1,pixels=0,capture;
    initial begin
        capture=$fopen("detour_demo.ppm","w");
        if (!capture) $fatal(1,"cannot open VGA capture");
        $fwrite(capture,"P3\\n640 480\\n255\\n");
        repeat(4) @(negedge clk);
        rst=0;
        forever begin
            @(posedge clk); #1;
            if(pixel_tick) begin
                if(frame_start) frame=frame+1;
                if(frame==1 && active) begin
                    if ((^{{red,green,blue}})===1'bx) $fatal(1,"unknown RGB");
                    $fwrite(capture,"%0d %0d %0d\\n",red*17,green*17,blue*17);
                    pixels=pixels+1;
                end
                if(frame==1 && pixel_x==799 && pixel_y==524) begin
                    if(pixels!=307200) $fatal(1,"incomplete VGA capture");
                    $fclose(capture);
                    $display("NAV_PREVIEW_PASS frames=2 pixels=307200");
                    $finish;
                end
            end
        end
    end
    initial begin #40000000; $fatal(1,"preview watchdog"); end
endmodule
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=("xsim", "iverilog"), default="xsim")
    parser.add_argument("--vivado-bin", type=Path,
                        default=Path("C:/Xilinx/Vivado/2024.1/bin") if os.name == "nt" else None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/navigation-preview")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    manifest, vector_hashes = verify_vector_bundle(ROOT / "test_vectors/navigation")
    scenario = next(case for case in manifest["cases"] if case["name"] == "detour")
    episodes = {}
    for bits in (8, 4):
        matching = [episode for episode in manifest["episodes"] if episode["case"] == "detour" and episode["bits"] == bits]
        if len(matching) != 1: raise ValueError("missing or duplicate detour episode")
        episodes[bits] = matching[0]
    sources = [Path(__file__).absolute(), ROOT / "rtl/maze_navigation_vga.v",
               ROOT / "scripts/run_navigation.py", ROOT / "scripts/stage_navigation.py"]
    before = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    before.update(vector_hashes)
    tb = out / "tb_navigation_preview.sv"
    tb.write_text(make_testbench(scenario, episodes), encoding="utf-8", newline="\n")
    hdl = [str(ROOT / "rtl/maze_navigation_vga.v"), str(tb)]
    if args.sim == "xsim":
        xvlog, xelab, xsim = (executable(name, args.vivado_bin) for name in ("xvlog", "xelab", "xsim"))
        version = run([xvlog,"-version"],out,"version.log",60)
        run([xvlog,"--sv",*hdl],out,"compile.log",args.timeout)
        run([xelab,"tb_navigation_preview","--snapshot","navigation_preview","--timescale","1ns/1ps"],
            out,"elaborate.log",args.timeout)
        output = run([xsim,"navigation_preview","--runall"],out,"simulation.log",args.timeout)
    else:
        iverilog, vvp = executable("iverilog"), executable("vvp")
        version = run([iverilog,"-V"],out,"version.log",60)
        run([iverilog,"-g2012","-Wall","-s","tb_navigation_preview","-o","preview.vvp",*hdl],
            out,"compile.log",args.timeout)
        output = run([vvp,"preview.vvp"],out,"simulation.log",args.timeout)
    if len(re.findall(r"^NAV_PREVIEW_PASS frames=2 pixels=307200$",output,re.MULTILINE)) != 1:
        raise RuntimeError("preview simulation did not report complete-frame success")
    frame = convert_frame(out / "detour_demo.ppm", out / "detour_demo.png")
    after = {name: sha(ROOT / name) for name in before}
    if before != after: raise RuntimeError("renderer, reference, or script changed during preview")
    report = {"format":"mazebits-reference-vga-preview-v1", "status":"pass",
              "description":"VGA rendering of verified reference final states; not a full-system CNN capture or board photograph.",
              "case":scenario,"simulator":args.sim,"tool_version":version.strip().splitlines()[:8],
              "episodes":[{key:episode[key] for key in ("bits","case","outcome","steps","final_position","trail_hex")}
                          for episode in episodes.values()],
              "frame":frame,"source_sha256":before,
              "artifacts":{name:{"sha256":sha(out/name),"bytes":(out/name).stat().st_size}
                           for name in ("tb_navigation_preview.sv","detour_demo.ppm","detour_demo.png")}}
    (out/"preview.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"PASS: {out / 'detour_demo.png'}",flush=True)


if __name__ == "__main__":
    main()
