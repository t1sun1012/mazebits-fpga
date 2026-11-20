"""Render the measured quantization report from frozen evidence and chart assets.

Run prepare_evaluation_report.py first. Requires reportlab and pypdf.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).absolute().parents[1]
EVIDENCE_COMMIT = '0bd19baa7dea709d8711b16bfb94b5be9aa4c523'
BASE_URL = f'https://github.com/t1sun1012/mazebits-fpga/blob/{EVIDENCE_COMMIT}/'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def make_report():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    from reportlab.lib.pagesizes import letter
    from pypdf import PdfReader

    metrics_path = ROOT / 'results/evaluation/metrics.json'
    m = json.loads(metrics_path.read_text(encoding='utf-8'))
    # Bind PDF numbers and chart pixels to the prepared evidence.
    for path, expected in m['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Source changed since report preparation: {path}')
    for name, record in m['charts'].items():
        if sha(ROOT / 'results/evaluation' / name) != record['sha256']:
            raise ValueError(f'Chart changed: {name}')
    b = json.loads((ROOT / 'results/navigation-structured/build/implementation.json').read_text())
    q = json.loads((ROOT / 'results/policy-improvement/quantization/comparison.json').read_text())
    board = m['physical_checks']
    assert board['status'] == 'required_checks_passed'
    a, c = m['models']['w8a8'], m['models']['w4a4']
    ar, cr = a['rollouts'], c['rollouts']
    ah, ch = a['hardware'], c['hardware']
    p = m['paired_outcomes']
    pct = lambda x: f'{100*x:.2f}%'
    n = lambda x: f'{x:,}'
    delta = 100 * (ar['success_rate'] - cr['success_rate'])
    memory_saving = 100 * (1 - ch['packed_parameter_bytes']/ah['packed_parameter_bytes'])
    bram_saving = 100 * (1 - ch['routed_cnn_resources']['bram_36k_equivalents']/ah['routed_cnn_resources']['bram_36k_equivalents'])
    lut_saving = 100 * (1 - ch['routed_cnn_resources']['luts']/ah['routed_cnn_resources']['luts'])

    out = ROOT / 'output/pdf'
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / 'MazeBits_Quantization_Evaluation.pdf'
    md_path = ROOT / 'docs/quantization-evaluation.md'
    font, bold = 'Helvetica', 'Helvetica-Bold'
    if Path('C:/Windows/Fonts/arial.ttf').exists():
        pdfmetrics.registerFont(TTFont('ReportArial', 'C:/Windows/Fonts/arial.ttf'))
        pdfmetrics.registerFont(TTFont('ReportArialBold', 'C:/Windows/Fonts/arialbd.ttf'))
        font, bold = 'ReportArial', 'ReportArialBold'
        pdfmetrics.registerFontFamily(font, normal=font, bold=bold, italic=font, boldItalic=bold)
    navy = colors.HexColor('#15283F')
    cyan = colors.HexColor('#007F9B')
    purple = colors.HexColor('#8446C5')
    muted = colors.HexColor('#526274')
    pale = colors.HexColor('#F2F6FA')
    styles = {
        'title': ParagraphStyle('title',fontName=bold,fontSize=30,leading=34,textColor=navy,spaceAfter=10),
        'h1': ParagraphStyle('h1',fontName=bold,fontSize=21,leading=26,textColor=navy,spaceAfter=13),
        'h2': ParagraphStyle('h2',fontName=bold,fontSize=11.5,leading=15,textColor=navy,spaceBefore=10,spaceAfter=5),
        'lead': ParagraphStyle('lead',fontName=bold,fontSize=13,leading=19,textColor=navy,spaceAfter=12),
        'body': ParagraphStyle('body',fontName=font,fontSize=9.5,leading=14,textColor=navy,spaceAfter=8),
        'small': ParagraphStyle('small',fontName=font,fontSize=8,leading=11.5,textColor=muted,spaceAfter=7),
        'cell': ParagraphStyle('cell',fontName=font,fontSize=8.5,leading=11.5,textColor=navy),
        'head': ParagraphStyle('head',fontName=bold,fontSize=8.5,leading=11.5,textColor=colors.white),
        'kicker': ParagraphStyle('kicker',fontName=bold,fontSize=9,leading=12,textColor=cyan,spaceAfter=10),
    }
    story, md = [], []
    def markdown_text(text):
        text = re.sub(r'<b>(.*?)</b>',r'**\1**',text)
        text = re.sub(r'<link href="([^"]+)">(.*?)</link>',r'[\2](\1)',text)
        return text.replace('&gt;','>').replace('&lt;','<').replace('&amp;','&')
    def para(text, style='body'):
        story.append(Paragraph(text,styles[style]))
        md.extend([markdown_text(text),''])
    def heading(text, level=1):
        story.append(Paragraph(text,styles['h1' if level==1 else 'h2']))
        md.extend([('#'*(level+1))+' '+text,''])
    def table(headers, rows, widths):
        cells=[[Paragraph(escape(str(x)),styles['head']) for x in headers]]
        cells += [[Paragraph(escape(str(x)),styles['cell']) for x in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,pale]),('LINEBELOW',(0,0),(-1,0),0.5,navy),('LINEBELOW',(0,-1),(-1,-1),0.5,colors.HexColor('#CFD8E3'))]))
        story.extend([t,Spacer(1,8)])
        md.append('| '+' | '.join(map(str,headers))+' |')
        md.append('| '+' | '.join(['---']*len(headers))+' |')
        for row in rows:md.append('| '+' | '.join(map(str,row))+' |')
        md.append('')
    def page():story.append(PageBreak())
    def source(label,title,path):
        return f'[{label}] <link href="{BASE_URL+path}">{title}</link>'

    para('EC311 / MAZEBITS / EVALUATION REPORT','kicker')
    story.append(Paragraph('Quantization<br/>Evaluation',styles['title']))
    md.extend(['# MazeBits quantization evaluation',''])
    para('Improved CNN: W8A8 versus W4A4 | November 19, 2026','small')
    para('4-bit saves memory. 8-bit provides more reliable navigation on structured mazes.','lead')
    para(f'On the same 4,800 validation routes, W8A8 reaches the goal in {pct(ar["success_rate"])} of runs and W4A4 in {pct(cr["success_rate"])}. W4A4 uses {memory_saving:.2f}% fewer packed parameter bytes and {bram_saving:.2f}% fewer CNN BRAM equivalents. Both engines take 32.17153 ms per decision at the implemented 100 MHz clock constraint.')
    chart=ROOT/'results/evaluation/comparison.png'
    iw,ih=ImageReader(str(chart)).getSize()
    story.extend([Image(str(chart),width=526,height=526*ih/iw),Spacer(1,6)])
    md.extend(['![Measured comparison](../results/evaluation/comparison.png)',''])
    para('Figure 1. Same improved checkpoint, two precisions. Quality is measured in the Python integer reference on validation data; resources come from routed Vivado hierarchy reports; latency comes from verified cycles at the routed clock constraint. Sources S1-S3.','small')
    heading('Main finding',2)
    para(f'The 4-bit route-success reduction is <b>{delta:.2f} percentage points</b>, or 716 fewer successes overall. The current serial engine has <b>no per-decision speedup</b> at 4 bits. This implementation demonstrates a storage/resource versus navigation-quality tradeoff.')
    para('Physical board tests confirm the improved 8-bit policy solves the previously failed 26-move detour. The 4-bit policy still loops on that map. All six required board checks pass by user report. These examples corroborate implementation behavior; aggregate accuracy comes from the validation set. [S4]','small')

    page(); heading('1. Navigation performance')
    para('Both models use identical layouts, starts, goals, action order, and termination rules. Actions are raw argmax outputs. No BFS planner, invalid-action repair, or visited-cell avoidance assists inference. BFS supplies labels and reference path lengths. [S2, S5]')
    table(['Metric / same 4,800 routes','W8A8','W4A4'],[
        ['Goal-reaching routes',f'{n(ar["success_count"])}/4,800',f'{n(cr["success_count"])}/4,800'],
        ['Route success',pct(ar['success_rate']),pct(cr['success_rate'])],
        ['Optimal-action accuracy / 27,084 states',pct(a['optimal_action_accuracy']),pct(c['optimal_action_accuracy'])],
        ['Blocked / loop / timeout','0 / 34 / 0','37 / 713 / 0'],
        ['Mean successful path length / BFS optimum',f'{ar["successful_mean_path_ratio"]:.5f}',f'{cr["successful_mean_path_ratio"]:.5f}'],
    ],[280,123,123])
    para('The FP32 source succeeds on 4,773/4,800 routes (99.44%). The 8-bit result is close to that source; most additional failures occur at 4 bits. Optimal-action accuracy counts any equally shortest action as correct. A single poor action can end a route, so high one-step accuracy does not guarantee high route success. Path ratios exclude failures: their denominators are 4,766 and 4,050 successful runs, respectively. Read them alongside success counts.','small')
    heading('Paired outcomes',2)
    table(['Both succeed','Only 8-bit succeeds','Only 4-bit succeeds','Both fail'],[[n(p['both_success']),n(p['only_8bit_success']),n(p['only_4bit_success']),n(p['both_fail'])]],[131.5]*4)
    para('The 10 routes solved only by W4A4 show that higher precision does not win on every individual maze. Quantization changes score rankings and can occasionally improve a particular trajectory. The aggregate favors W8A8.','small')
    heading('Where the gap appears',2)
    rows=[]
    for name,g in m['by_family'].items():
        label={'bernoulli':'Random obstacles','barrier':'Connected barriers','corridor':'Winding corridors'}.get(name,name)
        rows.append([label,n(g['episodes']),pct(g['models']['w8a8']['success_rate']),pct(g['models']['w4a4']['success_rate'])])
    for name,g in m['by_bfs_distance'].items():
        rows.append([f'BFS distance {name}',n(g['episodes']),pct(g['models']['w8a8']['success_rate']),pct(g['models']['w4a4']['success_rate'])])
    table(['Validation subgroup','Routes','W8A8 success','W4A4 success'],rows,[232,78,108,108])
    para('Family groups and distance groups are two views of the same 4,800 routes; do not add their denominators together. Layouts contribute 24 routes each, so episode outcomes are correlated. These are descriptive validation results, with no independent-episode confidence interval claimed.','small')

    page(); heading('2. Runtime and memory')
    heading('Timing measures equal work',2)
    para('<b>Latency = 3,217,153 cycles / 100,000,000 Hz = 32.17153 ms per decision.</b> Both precisions use the same serial schedule. Width reduction changes storage and logic size but not the number of scheduled operations. [S1, S3]')
    para('The displayed CYC HEX accumulates CNN cycles. It excludes the intentional 200 ms animation pause after nonterminal moves, button handling, VGA refresh, and other controller overhead. Fourteen decisions require 45,040,142 cycles, or 450.40142 ms of inference, for either model. Visible animation lasts longer.')
    pair = m['paired_successful_paths']
    p8 = pair['models']['w8a8']['successful_mean_steps'] * ah['nominal_inference_ms']
    p4 = pair['models']['w4a4']['successful_mean_steps'] * ch['nominal_inference_ms']
    para(f'On the same {pair["episodes"]:,} routes where both models succeed, mean inference-only route time is {p8:.5f} ms at 8 bits and {p4:.5f} ms at 4 bits. Their paths have equal length in {pair["same_length"]:,} cases; W8A8 is shorter in {pair["w8_shorter"]}, with {pair["extra_4bit_steps_total"]} extra W4A4 moves overall. This paired comparison avoids treating early failures as faster solutions.','small')
    table(['Storage / routed CNN metric','W8A8','W4A4'],[
        ['Packed weights / 16,624 values','16,624 bytes','8,312 bytes'],
        ['Packed biases / 132 signed 32-bit values','528 bytes','528 bytes'],
        ['Total packed parameters',f'{n(ah["packed_parameter_bytes"])} bytes',f'{n(ch["packed_parameter_bytes"])} bytes'],
        ['Two logical activation banks','2,048 bytes','1,024 bytes'],
        ['BRAM36 equivalents','9','5'],
        ['LUTs','632','524'],
        ['Registers','347','333'],
        ['DSP blocks','0','0'],
    ],[280,123,123])
    para(f'W4A4 saves {memory_saving:.2f}% of packed parameter storage, {bram_saving:.2f}% of CNN BRAM allocation, and {lut_saving:.2f}% of CNN LUTs in this combined routed design. Its nominal speedup is 1.00x. Biases remain 32-bit, so total parameter storage does not halve exactly.')
    heading('Payload, working memory, and physical allocation',2)
    para('Packed parameter bytes describe the binary model payload. Activation-bank bytes describe intermediate tensors only. They exclude score accumulators, inputs, control registers, and memory-allocation overhead. BRAM equivalents describe the physical blocks allocated by Vivado: the 8-bit core uses eight RAMB36 plus two RAMB18 blocks; the 4-bit core uses four RAMB36 plus two RAMB18 blocks.')
    para('ASCII .mem file size and full .bit file size are unsuitable measures of model storage. The bitstream also configures routing and the rest of the device. Zero DSP blocks means synthesis implements arithmetic in logic; multiplication still occurs.','small')
    heading('Whole-design timing closure',2)
    para('The combined two-engine VGA/navigation build uses <b>2,176 LUTs, 1,695 registers, 14.5 BRAM36 equivalents, and 0 DSP blocks</b>. At a 10 ns clock constraint, routed setup slack is +0.603 ns and hold slack is +0.042 ns, with zero DRC violations. Per-CNN counts above exclude surrounding logic; hierarchy attribution can depend on synthesis optimization. [S3]')
    para('These are implemented-design results and nominal cycle-derived timing. They are not an independent oscillator-frequency measurement, energy measurement, or speed comparison against optimized CPU inference. A failed route that stops earlier is not faster successful navigation.','small')

    page(); heading('3. FPGA verification and board results')
    para('The improved integer models were verified before programming. The final bitstream was built from the same generated CNN HDL and memory-initialization bytes as the passing navigation simulation, checked again after staging integrity hardening. [S6, S7]')
    table(['Verification layer','Recorded coverage'],[
        ['Python tests','153 passed'],
        ['Each CNN RTL core','8 fixtures; 133,160 intermediate/output words; repeat and protocol checks'],
        ['Arithmetic edge cases, per core','13 rounding cases; 3 signed/tie argmax cases'],
        ['Complete dual-CNN navigation','7 scenarios; 14 episodes; 157 decisions; 628 exact signed-score matches'],
        ['Controls and VGA','202 controller checks; 34 integrated checks; 3 complete VGA frames'],
        ['Original-checkpoint regression','4 scenarios; 8 episodes; 60 decisions; 240 scores'],
    ],[176,350])
    heading('User-reported tests on the improved bitstream',2)
    table(['Physical maze','W8A8','W4A4'],[
        ['Open maze','GOAL / 014','GOAL / 014'],
        ['Single-column detour','GOAL / 021','LOOP / 008'],
        ['Previously failed long detour','GOAL / 026','LOOP / 004'],
    ],[280,123,123])
    para('Each result includes the expected displayed cycle count: open 02AF420E for both; detour 0406E315 / 0188B808; long detour 04FC561A / 00C45C04. Restart during movement, cancellation on returning to editing, and start equal to goal with zero attempts/cycles also pass. [S4]')
    para('The three maze runs and three controls complete the required physical checklist. Other reference layouts have simulation results but remain physically untested. These observations were reported by the user; the improved-build checks have no newly supplied monitor photographs in this record.','small')
    heading('Interpretation',2)
    para('The long-detour result demonstrates that the improved 8-bit model handles a layout the original model failed. The 4-bit loop is consistent with its exact integer reference, rather than an observed arithmetic mismatch in the exercised RTL tests. Full correctness for every possible editable maze is not established by this finite suite.')
    para('Known demonstration maps were excluded from generated training splits, including their rotations/reflections. They were nevertheless used during development, so these examples do not provide an unbiased estimate of generalization. The validation suite supplies the broader comparison.','small')

    page(); heading('4. Methodology, limits, and conclusion')
    para('<b>Controlled comparison.</b> Both models are post-training quantizations of one improved FP32 checkpoint. Architecture: eight 3x3 convolution/ReLU layers with 16 channels, then a 1x1 four-action head; 16,624 weights and 132 biases. Inputs are three binary 8x8 planes for walls, current agent, and goal. Weights are signed 8/4-bit, hidden activations unsigned 8/4-bit; biases, accumulators, and final scores remain signed 32-bit.')
    para('Calibration uses 4,096 training states with seed 312. The structured dataset has 1,200 training layouts, 200 validation layouts, and 200 reserved-test layouts. The main evaluation covers all 27,084 labeled validation states and 4,800 fixed routes. The FP32 checkpoint was selected using this validation set, so these numbers are development estimates. Reserved test sets have not been evaluated. [S2, S5]')
    para('<b>Distribution matters.</b> On the original random-layout validation distribution, the same improved models score 3,600/3,600 (100.00%) at 8 bits and 3,557/3,600 (98.81%) at 4 bits. The larger gap on the structured distribution shows that one easy-maze score cannot establish robustness across maze types. [S8]')
    para('<b>Conclusion.</b> Use W8A8 as the current navigation model when route reliability is the priority. Keep W4A4 as the measured resource-saving comparison: it reduces model payload and FPGA memory substantially, while losing reliability most clearly on winding corridors and long routes. The present architecture does not turn the smaller width into lower latency.')
    para('<b>Next evaluation.</b> Freeze the selected models and evaluate the reserved test partition before making final generalization claims. Further changes to W4A4 should be evaluated with the same paired route protocol and exact RTL checks. Power and optimized CPU latency require separate measurements.','small')
    heading('Evidence and reproduction',2)
    sources=[
        ('S1','Matched-checkpoint benchmark','results/navigation-structured/benchmark.json'),
        ('S2','Integer validation and calibration provenance','results/policy-improvement/quantization/comparison.json'),
        ('S3','Routed resources, timing, DRC, and build identity','results/navigation-structured/build/implementation.json'),
        ('S4','Physical board-test observations','results/navigation-structured/board-test.json'),
        ('S5','Dataset audit and evaluation protocol','results/dataset-v2/README.md'),
        ('S6','Core and complete-navigation verification','results/navigation-structured/README.md'),
        ('S7','Final generated-input equivalence','results/navigation-structured/final-stager-equivalence.json'),
        ('S8','Cross-distribution improvement evidence','results/policy-improvement/README.md'),
    ]
    for label,title,path in sources:para(source(label,title,path),'small')
    para(f'Evidence snapshot: {EVIDENCE_COMMIT[:12]}. Final bitstream SHA-256: {board["bitstream_sha256"][:32]}<br/>{board["bitstream_sha256"][32:]}. Full hashes, paired counts, subgroup counts, and figure identities are in results/evaluation/metrics.json.','small')
    para('Reproduce from the repository root: run scripts/prepare_evaluation_report.py, then scripts/render_evaluation_report.py. The first uses existing evidence plus validation-layout metadata and matplotlib; the second requires reportlab and pypdf. No retraining, new inference, or reserved-test access is performed to compile this report.','small')

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#D5DEE8'))
        canvas.line(43,37,569,37)
        canvas.setFont(font,7.5);canvas.setFillColor(muted)
        canvas.drawString(43,25,'MAZEBITS | IMPROVED CHECKPOINT | VALIDATION + FPGA EVIDENCE')
        canvas.drawRightString(569,25,f'{doc.page} / 5')
        canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf),pagesize=letter,rightMargin=43,leftMargin=43,topMargin=42,bottomMargin=49,title='MazeBits Quantization Evaluation: W8A8 versus W4A4',author='MazeBits project team',subject='Measured comparison of improved integer CNN policies on FPGA')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(str(pdf))
    if len(reader.pages)!=5:
        raise RuntimeError(f'Expected five report pages, rendered {len(reader.pages)}; inspect layout before publication')
    text='\n'.join(page.extract_text() for page in reader.pages)
    for token in ['99.29%','84.38%','32.17153','4,040','726','10','153','26']:
        assert token in text,token
    md_path.write_text('\n'.join(md),encoding='utf-8',newline='\n')
    record={'format':'mazebits-report-render-v1','source_evidence_commit':EVIDENCE_COMMIT,'metrics_sha256':sha(metrics_path),'generator_sha256':sha(Path(__file__)),'pdf':{'path':pdf.relative_to(ROOT).as_posix(),'sha256':sha(pdf),'pages':len(reader.pages)},'markdown':{'path':md_path.relative_to(ROOT).as_posix(),'sha256':sha(md_path)},'text_checks':'pass','visual_review':'pending'}
    (ROOT/'results/evaluation/report-render.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Created {pdf} ({len(reader.pages)} pages) and {md_path}')

if __name__=='__main__':make_report()
