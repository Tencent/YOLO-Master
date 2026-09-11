#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 结项汇报 PPT v2 — 10 页 16:9 论点式框架(2026-09-07, stage4 定稿数字)。
框架: 封面(主张) → 协议 → 结果总览 → 发现①性价比 → 发现②稳定性 → 发现③Planner
      → 决策建议 → 证据链/复现 → 边界声明 → 收尾。
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------- 画布 16:9 ----------
SW, SH = 13.333, 7.5
prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)

# ---------- 色板: 工业探伤(炭蓝底 × 探照青 × 警示琥珀) ----------
INK   = RGBColor(0x14, 0x22, 0x2E)   # 主墨
PANEL = RGBColor(0xF2, 0xF6, 0xF8)   # 近白面板
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
DARK  = RGBColor(0x0F, 0x1B, 0x26)   # 封面底
DARK2 = RGBColor(0x1B, 0x2A, 0x39)   # 卡底
SLATE = RGBColor(0x6E, 0x7B, 0x87)   # 次要字
STEEL = RGBColor(0x2E, 0x4B, 0x63)   # full_sft 身份
CYAN  = RGBColor(0x14, 0x9E, 0x8C)   # vpeft 身份 / 主强调
AMBER = RGBColor(0xD9, 0x8A, 0x2B)   # frozen 身份 / 警示
RED   = RGBColor(0xD5, 0x4B, 0x3F)   # 异常
GREEN = RGBColor(0x2E, 0x8B, 0x63)   # 通过
WHT   = RGBColor(0xFF, 0xFF, 0xFF)
GRID  = RGBColor(0xDC, 0xE4, 0xEA)

FONT = "Microsoft YaHei"

def _style(r, size, color, bold=False, italic=False):
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = FONT
    rPr = r._r.get_or_add_rPr()
    ea = rPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ea')
    if ea is None:
        from lxml import etree
        ea = etree.SubElement(rPr, '{http://schemas.openxmlformats.org/drawingml/2006/main}ea')
    ea.set('typeface', FONT)

def rect(slide, x, y, w, h, fill=None, line=None, lw=0.75, rounded=False):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        try: shp.adjustments[0] = 0.06
        except Exception: pass
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(lw)
    return shp

def text(slide, x, y, w, h, paras, align='l', anchor='t', wrap=True):
    """paras: list of paragraphs; each = list of (txt, size, color, bold) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE, 'b': MSO_ANCHOR.BOTTOM}[anchor]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[align]
        if isinstance(para, tuple) and isinstance(para[0], str):
            para = [para]  # 单 run 直接作段落
        for run in para:
            if len(run) == 4:
                txt, size, color, bold = run
            else:
                txt, size, color = run; bold = False
            r = p.add_run(); r.text = txt; _style(r, size, color, bold)
    return tb

def P(text, size=12, color=INK, bold=False):
    return [(text, size, color, bold)]

def num_fmt(v, nd=3):
    return f"{v:.{nd}f}"

# ---------- 页脚 / 标题带 ----------
PAGENO = [0]
def header(slide, tag, title, accent=CYAN):
    """tag: '01 · 协议' 之类；title=断言句。返回标题底部坐标。"""
    # 左上 tag chip
    rect(slide, 0.42, 0.46, 1.24, 0.36, fill=accent, rounded=True)
    text(slide, 0.42, 0.49, 1.24, 0.30, [P(tag, 9.5, WHT, True)], align='c', anchor='m')
    # 断言标题(可两行, 与内容区留白)
    text(slide, 1.86, 0.36, 11.1, 0.82, [P(title, 21, INK, True)], anchor='m')

def footer(slide, idx, note):
    rect(slide, 0.55, 7.04, 12.25, 0.012, fill=GRID)
    text(slide, 0.55, 7.10, 10.5, 0.3, [P("证据: " + note, 8.5, SLATE)])
    text(slide, 12.0, 7.10, 0.8, 0.3, [P(f"{idx:02d}", 9, SLATE, True)], align='r')

def new_content():
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, SW, SH, fill=PAPER)
    return s

def stat(slide, x, y, w, h, big, label, note, bigcolor=CYAN, box=False):
    if box:
        rect(slide, x, y, w, h, fill=PANEL, rounded=True)
    text(slide, x + 0.22, y + 0.16, w - 0.44, 0.9, [P(big, 40, bigcolor, True)])
    text(slide, x + 0.22, y + 1.05, w - 0.44, 0.42, [P(label, 12.5, INK, True)])
    if note:
        text(slide, x + 0.22, y + 1.44, w - 0.44, 0.6, [P(note, 9.5, SLATE)])

def chip(slide, x, y, txt, fill, txtc, size=11, w=None, h=0.34):
    if w is None:
        lat = sum(1 for ch in txt if ord(ch) <= 0x2E80)
        cjk = len(txt) - lat
        w = 0.30 + (cjk * 1.02 + lat * 0.56) * size / 72.0
    rect(slide, x, y, w, h, fill=fill, rounded=True)
    text(slide, x, y, w, h, [P(txt, size, txtc, True)], align='c', anchor='m')
    return w

# =====================================================================
# Slide 1  封面(dark)
# =====================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
rect(s, 0, 0, SW, SH, fill=DARK)
rect(s, 0, 0, SW, 0.10, fill=CYAN)
rect(s, 0.55, 0.85, 0.10, 0.72, fill=CYAN)
text(s, 0.90, 0.82, 11.0, 0.4, [P("腾讯犀牛鸟精英培养计划 · C3 课题  |  结项汇报", 14, RGBColor(0xAF, 0xC2, 0xD2))])
text(s, 0.55, 1.95, 12.2, 1.0, [
    [("V-PEFT", 48, CYAN, True), ("  × 小样本工业缺陷检测", 48, WHT, True)],
])
text(s, 0.57, 3.35, 11.6, 0.85, [P("以 4.15% 的可训参数逼近全量微调——差距与稳定性边界的 3-seed 受控测量", 18, RGBColor(0xD8, 0xE2, 0xEA))])
# chips(顺序排布, 自动宽度)
cx = 0.57
for c in ["YOLO-Master EsMoE-N · 全参 2.8M", "NEU-DET 1800 张 · DeepPCB",
          "3 策略 × 3 seed × 同预算", "21 训练单元 · 全部 exit 0"]:
    cx += chip(s, cx, 4.75, c, DARK2, RGBColor(0xD8, 0xE2, 0xEA), size=11) + 0.26
text(s, 0.57, 6.7, 8.0, 0.4, [P("stage1→5 全链路证据 · 逐 PR 可评审 · reproduction/ 一键复现", 12, RGBColor(0x8F, 0xA3, 0xB5))])
text(s, 10.6, 6.7, 2.2, 0.4, [P("2026-09-07", 12, RGBColor(0x8F, 0xA3, 0xB5))], align='r')

# =====================================================================
# Slide 2  协议
# =====================================================================
s = new_content()
header(s, "01 · 协议", "先定一场“没有借口的对照”：同预算、3 seed、唯一变量 = 策略", accent=CYAN)
# 左: 问题面板
rect(s, 0.55, 1.32, 7.55, 1.92, fill=RGBColor(0xE3, 0xF3, 0xF0), rounded=True)
text(s, 0.82, 1.48, 7.0, 0.32, [P("要回答的问题", 12.5, RGBColor(0x0E, 0x6E, 0x62), True)])
text(s, 0.82, 1.84, 7.0, 1.32, [
    [("同等训练预算下，", 13.5, INK), ("V-PEFT", 13.5, CYAN, True),
     ("(LoRA + planner 自动放置) vs “全量微调 / 冻结主干”——在 2.8M 小模型 + 小样本工业缺陷数据上，精度守住多少、代价几何？", 13.5, INK)]
])
# 左: 公平协议 bullets
rect(s, 0.55, 3.40, 7.55, 3.28, fill=PANEL, rounded=True)
text(s, 0.82, 3.56, 7.0, 0.32, [P("公平协议（怎么保证结论不是器材或运气）", 12.5, STEEL, True)])
pro = [
    ("唯一变量 = 策略", "同预算 epochs=100 · batch=8 · imgsz=640 · amp=false", INK),
    ("不只信一个 seed", "seed 824/2024/777 各 3 个；另加 3 个补充单元做稳定性/确定性验证", INK),
    ("指标口径防误导", "best-mAP50(取各 epoch 最优, 非末行) · 显存取日志峰值 · 时长/单元", INK),
    ("执行可中断可审计", "done 标记 + resume, 8×A800 共享排队, 21/21 exit 0", INK),
]
yy = 3.98
for t, d, c in pro:
    text(s, 1.0, yy, 6.9, 0.52, [
        [("▸ ", 11, CYAN, True), (t + "：", 12, c, True)],
        P(d, 11, SLATE),
    ])
    yy += 0.61
# 右: 三策略卡
card_x, card_w = 8.5, 4.28
cards = [("full_sft", "全量微调(基线)", "2,813,626 可训(100%), 整网反传", STEEL),
         ("vpeft", "V-PEFT + planner", "adapter 116,736(4.15%); +解冻 head 465,250(16.5%)", CYAN),
         ("frozen_backbone", "freeze=11 冻结主干", "1,906,822 可训(67.8%)", AMBER)]
yy = 1.42
for name, sub, desc, c in cards:
    rect(s, card_x, yy, card_w, 1.72, fill=None, line=GRID, lw=1.2, rounded=True)
    rect(s, card_x, yy, 0.14, 1.72, fill=c)
    text(s, card_x + 0.34, yy + 0.14, card_w - 0.6, 0.4, [
        [(name, 16, INK, True), ("   " + sub, 11, SLATE)]
    ])
    text(s, card_x + 0.34, yy + 0.56, card_w - 0.66, 1.0, [P(desc, 10.5, c, True)])
    yy += 1.77
# 协议面板底部注(每策略同款训练数据)
text(s, 0.82, 6.40, 7.0, 0.26, [P("训练数据: 每(策略×seed)使用相同 k=10 全量子集(184 张), 唯一差异=可训练范围与放法", 9.5, SLATE)])
footer(s, 2, "stage1_env_data/README + stage3_matrix/matrix.json(21 单元定义)")

# =====================================================================
# Slide 3  结果总览
# =====================================================================
s = new_content()
header(s, "02 · 答案轮廓", "预算相同 → 参数省 95.8%、精度只低 ~0.07(NEU)/~0.11(PCB)", accent=STEEL)
# 左: 主表
rows = [
    ["数据集", "策略", "mAP50 mean±sd", "seed 明细(best)", "相对 full"],
    ["NEU", "full_sft", "0.767 ± 0.014", "0.752 / 0.780 / 0.768", "—"],
    ["NEU", "vpeft", "0.694 ± 0.052*", "0.691 / 0.369⚠ / 0.746 / 0.644", "−0.073"],
    ["NEU", "frozen", "0.694 ± 0.041*", "0.685 / 0.210⚠ / 0.659 / 0.739", "−0.073"],
    ["PCB", "full_sft", "0.989 ± 0.001", "0.989 / 0.989 / 0.990", "—"],
    ["PCB", "vpeft", "0.881 ± 0.072", "0.864 / 0.819 / 0.961", "−0.108"],
    ["PCB", "frozen", "0.769 ± 0.151", "0.639⚠ / 0.891 / 0.907 / 0.639⚠", "−0.220"],
]
tx, ty, tw = 0.55, 1.42, 7.6
colw = [1.05, 1.25, 1.7, 2.6, 1.0]
rowh = [0.42, 0.55, 0.55, 0.55, 0.42, 0.55, 0.55]
g = s.shapes.add_table(len(rows), 5, Inches(tx), Inches(ty), Inches(tw), Inches(sum(rowh))).table
g.first_row = False; g.horz_banding = False
for j, w in enumerate(colw):
    g.columns[j].width = Inches(w)
for i, w in enumerate(rowh):
    g.rows[i].height = Inches(w)
for i, row in enumerate(rows):
    hl = row[1] == "vpeft"
    for j, val in enumerate(row):
        cell = g.cell(i, j)
        cell.margin_left = cell.margin_right = Inches(0.07)
        cell.margin_top = cell.margin_bottom = Inches(0.015)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tfc = cell.text_frame; tfc.word_wrap = True
        p = tfc.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j in (0, 1, 3) else PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(val)
        if i == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = INK
            _style(r, 11.5, WHT, True)
        else:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xEA, 0xF1, 0xF6) if hl else (PAPER if i % 2 else PANEL)
            if j == 1:
                cid = {"full_sft": STEEL, "vpeft": CYAN, "frozen": AMBER}[row[1]]
                _style(r, 12, cid, True)
            elif "⚠" in str(val):
                _style(r, 10.5, AMBER, False)
            else:
                _style(r, 11, INK, j == 4)
text(s, tx, ty + sum(rowh) + 0.14, tw, 0.85, [
    P("* NEU vpeft/frozen 为稳健口径(剔除 1 个偶发 seed, n=3; 全 4-seed 均值 0.612/0.573)", 9.5, SLATE),
    P("⚠ = 收敛异常(seed 偶发或确定性动力学, 详见第 5 页)", 9.5, SLATE),
])
# 右: 三张大卡(可训参数 / 显存 / 时长成本) + 引导句
rx, rw = 8.4, 4.38
rect(s, rx, 1.42, rw, 5.2, fill=PANEL, rounded=True)
text(s, rx + 0.3, 1.6, rw - 0.6, 0.32, [P("“省下来的钱”长什么样", 13.5, STEEL, True)])
stat(s, rx + 0.3, 2.05, rw - 0.6, 1.35, "−95.8%", "可训参数: 2.81M → 0.117M", None, bigcolor=CYAN)
stat(s, rx + 0.3, 3.55, rw - 0.6, 1.35, "−1.2 G", "峰值显存: 5.3G → 4.2G (A800)", None, bigcolor=GREEN)
stat(s, rx + 0.3, 5.05, rw - 0.6, 1.3, "≈ 持平", "时长/单元(数据管线为瓶颈)", None, bigcolor=SLATE)
footer(s, 3, "stage4_analysis/README.md 表 1-3 + comparison_tables.md")

# =====================================================================
# Slide 4  发现① 性价比
# =====================================================================
s = new_content()
header(s, "03 · 发现①", "性价比成立: 参数降 95.8%、显存省 1.2G, 精度代价在十分位", accent=CYAN)

def minibar(s, x0, title_y, title, plot_top, plot_h, w, vmax, items, vscale=True):
    text(s, x0, title_y, w, 0.32, [P(title, 12.5, INK, True)])
    base = plot_top + plot_h
    rect(s, x0, base, w, 0.013, fill=SLATE)
    for k in range(3):
        gy = base - plot_h * (k + 1) / 3.0
        rect(s, x0, gy, w, 0.006, fill=GRID)
    nb = len(items)
    bw = w / nb * 0.46
    for i, (lab, val, c) in enumerate(items):
        bh = max(plot_h * val / vmax, 0.08)
        cx = x0 + w / nb * (i + 0.5)
        rect(s, cx - bw / 2, base - bh, bw, bh, fill=c)
        if bh > 0.34:  # 值标放柱顶内侧
            text(s, cx - bw / 2, base - bh + 0.03, bw, 0.3, [P(num_fmt(val), 11, WHT, True)], align='c')
        else:
            text(s, cx - 0.6, base + 0.02, 1.2, 0.3, [P(num_fmt(val), 10, c, True)], align='c')
        text(s, cx - 0.7, base + 0.34, 1.4, 0.26, [P(lab, 10, SLATE)], align='c')

minibar(s, 0.65, 1.42, "NEU-DET · mAP50(best, 稳健口径, n=3)", 1.9, 2.0, 5.5, 0.9,
        [("full", 0.767, STEEL), ("vpeft", 0.694, CYAN), ("frozen", 0.694, AMBER)])
minibar(s, 0.65, 4.55, "DeepPCB · mAP50(best, 3 seed)", 5.0, 1.42, 5.5, 1.1,
        [("full", 0.989, STEEL), ("vpeft", 0.881, CYAN), ("frozen", 0.769, AMBER)])
# 右: 判读
rx = 6.55
rect(s, rx, 1.5, 6.25, 5.3, fill=PANEL, rounded=True)
text(s, rx + 0.3, 1.72, 5.7, 0.35, [P("读法(逐项检验)", 13.5, STEEL, True)])
reads = [
    ("① 参数", "adapter 只占 4.15%(116,736/2.81M), 为最低约束档; 精度反而高于同预算的冻结(NEU 稳健口径持平, PCB 高 0.11)", STEEL),
    ("② 精度", "NEU 差 −0.07; PCB 差 −0.11 —— 都在 0.1 以内, 且 PCB 全量已趋饱和(0.989), 压差空间本就小", INK),
    ("③ 显存", "峰值 −1.2G(4.2G vs 5.3G), 主要省在冻结 Backbone 梯度; 边缘部署若以显存为约束则红利直接", GREEN),
    ("④ 时长", "约持平: 瓶颈是数据管线而非反传, 小模型红利未兑现 —— 报告如实记录, 不把“快”当卖点", AMBER),
    ("⑤ 代价", "代价不是参数而是稳定性(seed 偶发), 见下一页 —— 这是取舍的真正位置", RED),
]
yy = 2.2
for t, d, c in reads:
    text(s, rx + 0.3, yy, 5.7, 0.85, [
        [(t + "   ", 12.5, c, True)],
        P(d, 11.5, INK),
    ])
    yy += 0.92
footer(s, 4, "stage3_matrix/runs/*/summary.json + stage4/README.md 表 2(配对差)")

# =====================================================================
# Slide 5  发现② 稳定性(dot-spread)
# =====================================================================
s = new_content()
header(s, "04 · 发现②", "稳定性是隐藏的第三维度: 全量全稳, vpeft/frozen 各有 1/4 seed 收敛异常", accent=AMBER)
# 左: dot spread
rect(s, 0.55, 1.4, 6.6, 4.3, fill=PANEL, rounded=True)
text(s, 0.82, 1.55, 6.0, 0.32, [P("NEU-DET 逐 seed 分布(best mAP50, 越靠右越好)", 12.5, INK, True)])
srow = [("full_sft", [0.752, 0.780, 0.768], STEEL),
        ("vpeft  ", [0.691, 0.369, 0.746, 0.644], CYAN),
        ("frozen ", [0.685, 0.210, 0.659, 0.739], AMBER)]
sp_y0, sp_h = 2.15, 0.85
vm = 0.9
for i, (lab, vals, c) in enumerate(srow):
    yy = sp_y0 + i * (sp_h + 0.12)
    text(s, 0.85, yy, 1.0, 0.3, [P(lab, 11, c, True)])
    for v in vals:
        x = 1.95 + (v / vm) * 4.9
        fill = RED if v < 0.5 else c
        rect(s, x - 0.055, yy - 0.055, 0.11, 0.11, fill=fill)
    text(s, 5.3, yy, 1.7, 0.3, [P("", 10, SLATE)])
# axis
rect(s, 1.95, sp_y0 + 3 * sp_h + 0.1, 4.9, 0.012, fill=SLATE)
text(s, 1.9, sp_y0 + 3 * sp_h + 0.16, 0.6, 0.25, [P("0", 8.5, SLATE)])
text(s, 5.75, sp_y0 + 3 * sp_h + 0.16, 1.2, 0.25, [P("0.9", 8.5, SLATE)])
text(s, 3.6, sp_y0 + 3 * sp_h + 0.16, 1.2, 0.25, [P("● 红 = 收敛异常", 8.5, RED)], align='c')
text(s, 0.82, 5.86, 6.2, 0.62, [
    [("全口径 sd: ", 11, INK, True), ("full 0.014  |  vpeft 0.167  |  frozen 0.245", 12, INK)],
    P("剔除 1 个偶发 seed 后: vpeft/frozen sd ≈ 0.04-0.05 —— 异常是“全有或全无”式的坏 seed", 10.5, SLATE),
])
# 右: 补充验证表
rx = 7.4
rect(s, rx, 1.4, 5.4, 5.3, fill=None, line=GRID, lw=1.2, rounded=True)
text(s, rx + 0.28, 1.58, 4.9, 0.3, [P("验证实验(3 个补充单元, 全 exit 0)", 13, INK, True)])
vt = [
    ["补充单元", "结果(best)", "判定"],
    ["neu_vpeft_s2025", "0.644  正常", "s2024 停滞 = seed 偶发"],
    ["neu_frozen_s2025", "0.739  正常", "s2024 卡死 = seed 偶发"],
    ["pcb_frozen_s824b", "0.639@ep9 = s824 重跑逐一致", "震荡 = 确定性动力学"],
]
g = s.shapes.add_table(4, 3, Inches(rx + 0.28), Inches(2.0), Inches(4.85), Inches(2.1)).table
g.first_row = False; g.horz_banding = False
for j, w in enumerate([1.85, 1.55, 1.45]):
    g.columns[j].width = Inches(w)
for i in range(4):
    g.rows[i].height = Inches(0.5)
for i, row in enumerate(vt):
    for j, val in enumerate(row):
        cell = g.cell(i, j)
        cell.margin_left = Inches(0.05); cell.margin_right = Inches(0.05)
        cell.margin_top = cell.margin_bottom = Inches(0.01)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tfc = cell.text_frame; tfc.word_wrap = True
        p = tfc.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = val
        if i == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = INK; _style(r, 10.5, WHT, True)
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = PANEL if i % 2 else PAPER
            _style(r, 9.5, INK, False)
text(s, rx + 0.28, 4.35, 4.9, 2.2, [
    [("一句话: ", 12.5, RED, True), ("参数高效/冻结法的“坏 seed”是真实风险", 12.5, INK, True)],
    P("NEU 偶发率 ~1/4(各 1 例), 已由新 seed 验证为 seed 特异而非系统性失效;", 10.5, SLATE),
    P("PCB frozen@seed824 的震荡(高点落在 ep9)同 seed 重跑逐一致 → 是该 (seed, 冻结配置) 的训练动力学事实, 非采样噪声。", 10.5, SLATE),
    P("→ 单 seed 结论在参数高效法上不可信; 稳定性本身应作为报告维度。", 11, RED, True),
])
footer(s, 5, "stage3_matrix/README(补充单元表) + runs/*/train/*/results.csv 曲线")

# =====================================================================
# Slide 6  发现③ Planner
# =====================================================================
s = new_content()
header(s, "05 · 发现③", "Planner 可审计: 6/6 ACCEPT、决策跨 seed/数据稳定、护栏与缺陷如实", accent=STEEL)
# 左: 决策大卡
rect(s, 0.55, 1.4, 5.0, 5.3, fill=DARK, rounded=True)
text(s, 0.85, 1.62, 4.4, 0.3, [P("PLACEMENT · 6 个 vpeft 单元", 12, RGBColor(0xAF, 0xC2, 0xD2), True)])
stat(s, 0.85, 2.0, 4.4, 1.5, "6/6", "ACCEPT(无 ADAPT/REFUSE)", "planner 每次给出可用放置", bigcolor=CYAN)
text(s, 0.85, 3.78, 4.4, 0.3, [P("每单元一致决策", 11.5, RGBColor(0xAF, 0xC2, 0xD2), True)])
for i, (t, d) in enumerate([("81 targets", "非头层放置"), ("rank = 8", "base rank 不升"),
                              ("adapter 116,736", "4.15% 全参"), ("465,250 有效可训", "含 mismatch 解冻 head")]):
    xx = 0.85 + (i % 2) * 2.1
    yy = 4.14 + (i // 2) * 1.10
    rect(s, xx, yy, 1.95, 1.0, fill=DARK2, rounded=True)
    text(s, xx + 0.13, yy + 0.12, 1.7, 0.4, [P(t, 13, WHT, True)])
    text(s, xx + 0.13, yy + 0.52, 1.75, 0.4, [P(d, 8.5, RGBColor(0xAF, 0xC2, 0xD2))])
# 右: 护栏与缺陷
rx = 5.9
rect(s, rx, 1.4, 6.9, 5.3, fill=PANEL, rounded=True)
text(s, rx + 0.3, 1.62, 6.3, 0.35, [P("审计点 / 护栏(逐项可复核)", 13.5, STEEL, True)])
blocks = [
    ("ACCEPT 的语义", "状态与 predicted Δ 进入 audit; vpeft 接受后 legacy planner 显式跳过(ao/dco/mip 被接管)——日志 [V-PEFT] selected … ranks= 行可 grep", STEEL),
    ("类别 mismatch(80→6)", "检测头重初始化并解冻 → vpeft 有效可训 465,250(16.5%), 报告不混同 adapter 口径", INK),
    ("strict=True 不降级", "任何异常直接 raise(失败即报错), 证据链干净; 未静默 fallback", GREEN),
    ("缺陷已修 cap<8", "0.conv/routing_network.2/dfl.conv 容量 < 最小候选 rank(4): 原先求解器放行 → plan 校验抛错 → V-PEFT 静默降级 legacy; 修复分支已推 fork 并开上游 PR #279(C_cap 硬约束 + capacity_excluded 审计), 本报告数据仍是 exclude 规避口径未重跑 → 链接见末页①", GREEN),
    ("LOVO / ΔmAP 口径", "回归护栏系数与 REFUSE 阈值(−0.05)在源码; 实测 Δ 与 predicted_delta 对应关系记入 stage4/p2 笔记", SLATE),
]
yy = 2.12
for t, d, c in blocks:
    text(s, rx + 0.3, yy, 6.3, 0.9, [
        [("▪ ", 11, c, True), (t + "  ", 12, c, True)],
        P(d, 10.5, INK),
    ])
    yy += 0.90
footer(s, 6, "stage3 runs 日志 + stage4/planner_audit_summary.md + stage2/planner_solver_notes.md §3/§5")

# =====================================================================
# Slide 7  决策建议
# =====================================================================
s = new_content()
header(s, "06 · 怎么选", "按约束选策略: 显存/部署受限→vpeft; 要最高最稳精度→full; 冻结不划算", accent=GREEN)
cards = [
    ("场景 A", "显存 / 边缘受限", "选 vpeft", CYAN,
     "4.15% 参数 + 4.2G 峰值显存即达 full 九成以上精度(Δ ≤ 0.11); planner 零调参自动放置。",
     "避免: 精度敏感且无力复跑 seed 时"),
    ("场景 B", "算力充足 · 要最稳精度", "选 full_sft", STEEL,
     "NEU/PCB 均最优且三 seed 零异常(sd ≤ 0.014); 免去坏 seed 排查成本。",
     "代价: 显存多 1.2G, 参数 2.81M 全反传"),
    ("场景 C", "少调参的“冻结”方案", "不太推荐 frozen", AMBER,
     "可训参数只省 32%(67.8% 仍在训)却引入 seed 偶发与确定性震荡(PCB@824)。",
     "除非工具链强制冻结接口, 否则无性价比"),
]
cw, cg = 3.95, 0.2
xx = 0.55
for tag, scen, rec, c, why, avoid in cards:
    rect(s, xx, 1.5, cw, 4.7, fill=PANEL, rounded=True)
    rect(s, xx, 1.5, cw, 0.09, fill=c)
    text(s, xx + 0.26, 1.72, cw - 0.5, 0.3, [P(tag + " · " + scen, 11, SLATE)])
    text(s, xx + 0.26, 2.02, cw - 0.5, 0.55, [(rec, 21, c, True)])
    text(s, xx + 0.26, 2.62, cw - 0.52, 2.0, [
        P("理由", 11.5, INK, True),
        P(why, 11, SLATE),
    ])
    text(s, xx + 0.26, 4.9, cw - 0.52, 1.2, [
        P("注意", 11.5, RED, True),
        P(avoid, 10.5, SLATE),
    ])
    xx += cw + cg
text(s, 0.55, 6.45, 12.2, 0.5, [
    [("判定依据: ", 11, INK, True),
     ("stage4/README.md 表 1-2(mean/sd/配对差) · 表 3(参数/显存/时长) · 稳健口径定义见同文件", 10.5, SLATE)]
])
footer(s, 7, "stage4_analysis/README.md(结论判读节)")

# =====================================================================
# Slide 8  证据链与复现
# =====================================================================
s = new_content()
header(s, "07 · 证据链", "每条结论都给了评审一条可查证的路径(文件/git/复现包)", accent=STEEL)
ev = [
    ["结论", "证据文件(均已入库)", "内容"],
    ["精度与稳定性", "stage3_matrix/evidence_summary.csv · runs/*/results.csv", "21 单元 best-mAP50 + 逐 epoch 曲线 + 显存峰值"],
    ["统计(mean/sd/CI/配对差)", "stage4_analysis/comparison_tables.md", "3-seed 统计 + 稳健口径 + 四维表"],
    ["Planner 审计", "stage4_analysis/planner_audit_summary.md", "ACCEPT 分布 / 护栏 / ΔmAP 口径"],
    ["许可 / SHA / 环境", "stage1_env_data/README.md", "NEU-DET/DeepPCB 来源与哈希留档"],
    ["复现一键组装", "stage5_pr_delivery/reproduction/", "命令样例 · paths.env · 数据转换脚本 · 统计表"],
    ["缺陷修复(PR)", "分支 fix/vpeft-capacity-guard + PR #279", "C_cap 容量硬约束 + capacity_excluded 审计 + 未适配层冻结(2 commit · 链接见末页①)"],
]
tx, tw = 0.55, 7.7
colw = [2.1, 3.1, 2.5]
g = s.shapes.add_table(7, 3, Inches(tx), Inches(1.45), Inches(tw), Inches(4.1)).table
g.first_row = False; g.horz_banding = False
for j, w in enumerate(colw):
    g.columns[j].width = Inches(w)
for i in range(7):
    g.rows[i].height = Inches(0.54)
for i, row in enumerate(ev):
    for j, val in enumerate(row):
        cell = g.cell(i, j)
        cell.margin_left = Inches(0.07); cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.012)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tfc = cell.text_frame; tfc.word_wrap = True
        p = tfc.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = val
        if i == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = INK; _style(r, 11, WHT, True)
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = PANEL if i % 2 else PAPER
            _style(r, 10, INK, j == 0)
# 右: git 交付卡 + 复现包
rx = 8.55
rect(s, rx, 1.45, 4.25, 2.4, fill=DARK, rounded=True)
text(s, rx + 0.25, 1.65, 3.8, 0.3, [P("GIT · 8 本地 commit(逐 PR 四节)", 12, RGBColor(0xAF, 0xC2, 0xD2), True)])
for i, t in enumerate(["stage1 env+data", "stage2 server smoke (BF-01 修复)", "stage3 matrix + 21 单元", "stage4 stats + audit", "stage5 复现包 + 报告 + PPT"]):
    text(s, rx + 0.25, 2.0 + i * 0.36, 3.9, 0.3, [[("▸ ", 10.5, CYAN, True), (t, 10.5, WHT)]])
rect(s, rx, 4.05, 4.25, 2.65, fill=PANEL, rounded=True)
text(s, rx + 0.25, 4.25, 3.8, 0.3, [P("一键复现路径", 12.5, STEEL, True)])
for i, t in enumerate([
    "python stage5/assemble_reproduction.py",
    "→ reproduction/(命令/脚本/env/统计)",
    "三策略命令样例 → 修改 seed/device",
    "补充单元: run_supplement.py 自愈调度",
]):
    text(s, rx + 0.25, 4.6 + i * 0.5, 3.9, 0.45, [P(t, 10, INK, True)])
footer(s, 8, "git ls-files c3_work/ 全量证据 + stage5_pr_delivery/reproduction/")

# =====================================================================
# Slide 9  边界声明
# =====================================================================
s = new_content()
header(s, "08 · 边界", "边界声明: 这些数字为什么不能外推太远(评审先看这里)", accent=SLATE)
lim = [
    ("样本量", "n = 3-4 seed, CI 宽(±0.23~0.34 全口径); 偶发率 ~1/4 仅作警示, 不推总概率。", "n"),
    ("模型规模", "EsMoE-N 全参仅 2.8M: LoRA 相对全量的参数优势被小基数压缩; 结论不必然迁移到更大模型。", "M"),
    ("数据域", "两数据集(工业表面/PCB 缺陷), 分辨率与类别数有限; NEU-DET 镜像无显式 LICENSE(SHA-256 留档)。", "D"),
    ("预算形态", "同 epochs 预算下时长持平(数据管线瓶颈); 若改为“同 wall-clock”预算, 结论排序可能变化, 未测。", "T"),
    ("策略实现", "vpeft=当前 planner(rank 8/ao solver); frozen=freeze=11 单档, 未扫冻结深度。", "S"),
]
yy = 1.5
for tag, d, code in lim:
    rect(s, 0.55, yy, 12.25, 0.95, fill=PANEL, rounded=True)
    rect(s, 0.55, yy, 0.09, 0.95, fill=SLATE)
    text(s, 0.82, yy + 0.09, 1.2, 0.4, [P(tag, 13, INK, True)])
    text(s, 1.85, yy + 0.09, 9.9, 0.78, [P(d, 11.5, INK)])
    text(s, 11.9, yy + 0.22, 0.8, 0.5, [P(code, 16, SLATE, True)], align='c')
    yy += 1.02
text(s, 0.55, 6.66, 12.2, 0.3, [P("凡涉及外推(其他数据/更大模型/其他预算定义)均须重测; 本包已备好重测工具链。", 10.5, AMBER, True)])
footer(s, 9, "stage5/report_final.md §4 局限 与 stage4/README 口径红线")

# =====================================================================
# Slide 10  收尾(dark)
# =====================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
rect(s, 0, 0, SW, SH, fill=DARK)
rect(s, 0, SH - 0.10, SW, 0.10, fill=CYAN)
text(s, 0.7, 0.72, 12.0, 0.32, [P("一句话带走", 15, RGBColor(0x8F, 0xA3, 0xB5), True)])
text(s, 0.7, 1.12, 12.1, 0.92, [
    [("V-PEFT 在 2.8M 小模型上的答案: ", 24, WHT, True)],
    [("“低参数 / 低显存拿九成精度 —— 代价写进了 seed 分布里”", 24, CYAN, True)],
])
text(s, 0.72, 2.32, 12.0, 0.32, [P("三策略一句话结论(完整对照见 stage4)", 12.5, RGBColor(0xAF, 0xC2, 0xD2))])
sums = [
    ("full_sft", "0.767 / 0.989  |  sd ≤ 0.014", "最高且最稳的基线; 要精度就选它", STEEL),
    ("vpeft", "0.694* / 0.881  |  4.15% 参数", "省 95.8% 参数+1.2G 显存, 接受偶发风险", CYAN),
    ("frozen_backbone", "0.694* / 0.769  |  67.8% 可训", "省得少还添乱, 不推荐", AMBER),
]
xx = 0.7
for name, statline, desc, c in sums:
    rect(s, xx, 2.85, 3.9, 2.0, fill=DARK2, rounded=True)
    rect(s, xx, 2.85, 3.9, 0.07, fill=c)
    text(s, xx + 0.25, 3.05, 3.4, 0.4, [P(name, 16, WHT, True)])
    text(s, xx + 0.25, 3.5, 3.4, 0.4, [P(statline, 11.5, c, True)])
    text(s, xx + 0.25, 3.95, 3.45, 0.8, [P(desc, 10.5, RGBColor(0xAF, 0xC2, 0xD2))])
    xx += 4.05
text(s, 0.7, 5.02, 6.9, 0.32, [P("交付状态 / 下一步", 12.5, RGBColor(0xAF, 0xC2, 0xD2))])
todo = [
    ("① 修复分支已 push", "fix/vpeft-capacity-guard(2 commit); PR #279 待重填标题/正文"),
    ("② 证据包已 push", "fork lycyhrc/YOLO-Master:c3-vpeft-smoke(本地 commit 全部同步)"),
    ("③ 可选扩展", "k=5/10/50/100 档位曲线(数据就绪), GPU 低峰补跑"),
]
yy = 5.40
for t, d in todo:
    text(s, 0.7, yy, 7.0, 0.46, [[("▸ ", 11.5, CYAN, True), (t + "  ", 11.5, WHT, True), (d, 10.5, RGBColor(0xAF, 0xC2, 0xD2))]])
    yy += 0.48
rect(s, 7.95, 4.90, 4.68, 1.90, fill=DARK2, line=AMBER, lw=1.25, rounded=True)
text(s, 8.15, 5.02, 4.3, 0.3, [P("交付链接（② 待填 · 上传后替换）", 11, AMBER, True)])
text(s, 8.15, 5.38, 4.3, 0.32, [[("① 上游修复 PR #279  ", 9.5, WHT, True), ("已发起", 9.5, CYAN, True)]])
text(s, 8.15, 5.74, 4.3, 0.4, [P("Tencent/YOLO-Master/pull/279", 8.5, CYAN, True)])
text(s, 8.15, 6.24, 4.3, 0.32, [[("② 证据附件  ", 9.5, WHT, True), ("<<EVIDENCE_LINK_PLACEHOLDER>>", 9.5, CYAN, True)]])
text(s, 0.7, 6.92, 9.0, 0.25, [P("数字来源: stage3_matrix/runs(21 单元) · stage4_analysis(2026-09-07 定稿) · stage5 交付 2026-09-11", 10, RGBColor(0x8F, 0xA3, 0xB5))])

OUT = "C3_结项汇报_初稿.pptx"
prs.save(OUT)
print("saved", OUT, "| slides:", len(prs.slides._sldIdLst))
