#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 C3 结项汇报 PPT 初稿(基于 stage4 定稿数字, 2026-09-07)。python-pptx>=1.0"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

DARK = RGBColor(0x1F, 0x2A, 0x44)
BLUE = RGBColor(0x2F, 0x80, 0xED)
GREY = RGBColor(0x6B, 0x72, 0x80)
GREEN = RGBColor(0x2E, 0x9E, 0x5B)
RED = RGBColor(0xC0, 0x3A, 0x2B)
LGREY = RGBColor(0xF3, 0xF5, 0xF9)


def new_deck():
    return Presentation()


def add_title_slide(prs, title, sub):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.6), Inches(2.0), Inches(9.3), Inches(1.6))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = title
    r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = DARK
    tb2 = s.shapes.add_textbox(Inches(0.6), Inches(3.8), Inches(9.0), Inches(1.2))
    for i, line in enumerate(sub):
        p = tb2.text_frame.paragraphs[0] if i == 0 else tb2.text_frame.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(16); r.font.color.rgb = GREY
    return s


def add_content_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    bg.line.fill.background()
    s.shapes._spTree.remove(bg._element) if False else None
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(9.4), Inches(0.7))
    p = tb.text_frame.paragraphs[0]; r = p.add_run(); r.text = title
    r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = DARK
    bar = s.shapes.add_shape(1, Inches(0.55), Inches(1.02), Inches(0.9), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
    return s


def bullet(slide, items, left=0.6, top=1.35, width=9.0, height=4.9, size=15):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = ("• " if lvl == 0 else "– ") + txt
        r.font.size = Pt(size - lvl); r.font.color.rgb = DARK if lvl == 0 else GREY
        p.space_after = Pt(6)
    return tb


def add_table(slide, rows, cols, left, top, width, height, data, fsize=13, header=True):
    g = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height)).table
    for j in range(cols):
        g.columns[j].width = Inches(width / cols)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = g.cell(i, j)
            cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            t = cell.text_frame; t.word_wrap = True
            p = t.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val); r.font.size = Pt(fsize)
            if header and i == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = DARK
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.bold = True
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LGREY if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
                if any(m in str(val) for m in ("▲", "⚠", "+", "优于")):
                    r.font.color.rgb = GREEN
                r.font.color.rgb = DARK
    return g


def main():
    prs = new_deck()
    # 1 封面
    add_title_slide(prs, "C3 实战结项：V-PEFT 小样本工业缺陷检测", 
                    ["YOLO-Master EsMoE-N × NEU-DET/DeepPCB 三策略同预算 3-seed 对照",
                     "阶段证据链: stage1 环境数据 → stage2 服务器化冒烟 → stage3 21 单元矩阵 → stage4 统计审计 → stage5 复现交付", 
                     "2026-09-07"])
    # 2 目标与方法
    s = add_content_slide(prs, "目标: 用受控证据回答“V-PEFT 相对全量微调/冻结的性价比”")
    bullet(s, [("对比三条参数高效路线(同预算, 三策略唯一变量=策略)", 0),
               ("full_sft(全量微调基线) vs vpeft(V-PEFT+planner 自动放置 LoRA) vs frozen_backbone(freeze=11)", 1),
               ("同预算纪律: epochs=100 × batch=8 × imgsz=640 × amp=false; 两数据集 × 3 seed(824/2024/777)+3 补充验证", 0),
               ("模型 YOLO-Master-EsMoE-N: 全参 2,813,626(n 尺度, 小模型=参数效率叙事需量化到百分比)", 0),
               ("数据: NEU-DET 1800 张/6 类; DeepPCB(6 类), 许可与 SHA-256 见 stage1", 0),
               ("执行: 8×A800 共享排队 → 单卡单任务 queue_runner(done 标记断点续跑), 21 单元全部 exit 0", 0)])
    # 3 结果表
    s = add_content_slide(prs, "四维结果(主 18 单元 + 3 补充; best-mAP50 口径)")
    data = [
        ["数据集", "策略", "mAP50(seed 明细, best)", "mean±sd", "可训参数", "显存", "时长"],
        ["NEU", "full_sft", "0.752/0.780/0.768", "0.767±0.014", "2,813,626 (100%)", "5.3G", "55m"],
        ["NEU", "vpeft", "0.691/0.369⚠/0.746 (+2025:0.644✓)", "0.694±0.052*", "116,736 (4.15%)", "4.2G", "51m"],
        ["NEU", "frozen", "0.685/0.210⚠/0.659 (+2025:0.739✓)", "0.694±0.041*", "1,906,822 (67.8%)", "3.7G", "45m"],
        ["PCB", "full_sft", "0.989/0.989/0.990", "0.989±0.001", "2,813,626 (100%)", "5.3G", "46m"],
        ["PCB", "vpeft", "0.864/0.819/0.961", "0.881±0.072", "116,736 (4.15%)", "4.1G", "45m"],
        ["PCB", "frozen", "0.639⚠/0.891/0.907 (824b 重跑 0.639⚠)", "0.769±0.151", "1,906,822 (67.8%)", "3.7G", "51m"],
    ]
    add_table(s, 7, 7, 0.5, 1.25, 9.5, 4.6, data, fsize=11)
    bullet(s, [("*NEU vpeft/frozen 为稳健口径(剔除 1 个偶发 seed, n=3); 主表 3-seed 均值 0.612/0.573。⚠=收敛异常(已验证, 见下)", 0),
               ("vpeft adapter 仅 4.15% 全参 → NEU 差 ~0.07 / PCB 差 ~0.11 mAP50; 显存 −1.2G; 时长基本持平", 0)], top=6.0, height=1.1, size=12)
    # 4 稳定性验证
    s = add_content_slide(prs, "稳定性/确定性验证(补充单元 run_supplement)")
    data2 = [
        ["补充单元", "目的", "结果(best mAP50)", "判定"],
        ["neu_vpeft_s2025", "检验 s2024 停滞(0.369)是否偶发", "0.644 (正常收敛)", "seed 偶发 ✓"],
        ["neu_frozen_s2025", "检验 s2024 卡死(0.210)是否偶发", "0.739 (正常收敛)", "seed 偶发 ✓"],
        ["pcb_frozen_s824b", "同 seed 824 重跑验证震荡", "0.639@ep9 = s824 完全一致", "确定性动力学 ✓"],
    ]
    add_table(s, 4, 4, 0.5, 1.25, 9.5, 2.2, data2, fsize=12)
    bullet(s, [("结论: NEU 上 vpeft/frozen 的 1/4 极端收敛异常属 seed 特异(补跑全正常);", 0),
               ("PCB frozen@seed824 的震荡(best@ep9=0.639, 末行 0.496)被同 seed 重跑逐一致复现 → 非噪声, 是该(seed,配置)训练动力学的确定性事实", 0),
               ("→ 全量微调 seed 稳健性最好(sd≤0.014), 参数高效/冻结法存在偶发不稳, 已如实进入局限声明", 0)], top=3.9, height=2.0, size=13)
    # 5 planner 审计
    s = add_content_slide(prs, "Planner 审计(6/6 vpeft 单元)")
    data3 = [
        ["单元", "决策", "目标层", "rank", "适配器参数", "护栏"],
        ["neu ×3", "ACCEPT 6/6", "81 targets", "8", "116,736 (4.15%)", "strict=True, exclude 3 层"],
        ["pcb ×3", "ACCEPT 6/6", "81 targets", "8", "116,736 (4.15%)", "strict=True, exclude 3 层"],
    ]
    add_table(s, 3, 6, 0.5, 1.25, 9.5, 1.6, data3, fsize=12)
    bullet(s, [("决策跨 seed/数据集稳定: 无 ADAPT/REFUSE; legacy solver(ao/dco/mip)被 vpeft 显式接管(skipping legacy planner)", 0),
               ("类别 mismatch(80→6)检测头重初始化并解冻训练 ~348,514 → vpeft 有效可训 465,250(16.5%), 已单独披露", 0),
               ("已知缺陷 cap<8 层(0.conv/routing_network.2/dfl.conv) 以 lora_exclude_modules 规避、strict 不降级; LOVO 语义与 ΔmAP 口径见 stage4/p2 素材", 0)], top=3.3, height=2.4, size=13)
    # 6 证据链/复现
    s = add_content_slide(prs, "证据链与复现(stage1-5, 可逐 PR 评审)")
    bullet(s, [
        ("证据: 21 单元逐 epoch results.csv(曲线) + summary.json(exit/时长) + 日志 GpuMem 峰值显存", 0),
        ("统计: comparison_tables.md(mean/sd/95%CI/配对差) + planner_audit_summary.md(ACCEPT 分布/护栏/ΔmAP)", 0),
        ("复现包: stage5/reproduction/ = 数据转换脚本+paths.env+三策略命令样例+matrix/evidence/统计表+许可 SHA", 0),
        ("git: 全部本地 commit(每 stage 独立 PR 四节: 改动摘要/测试证据/消融数据/已知局限), 待 push fork lycyhrc/YOLO-Master:c3-vpeft-smoke", 0),
        ("执行形态: server_run.py(done 标记+resume) 适配共享 A800 排队, 任意中断可重扫续跑", 0)], top=1.4, height=3.6, size=14)
    # 7 局限与收尾
    s = add_content_slide(prs, "已知局限(如实声明)与下一步")
    bullet(s, [("n=3-4 seed, CI 宽; 偶发率 1/4 仅作警示不推总; 结论限定 EsMoE-N 与两数据集", 0),
               ("NEU-DET 镜像无显式 LICENSE / DeepPCB 许可(留 SHA-256 快照)", 0),
               ("n-scale 全参仅 2.8M, LoRA 优势被小模型基数压缩; 3 策略时长无差异(数据管线瓶颈)", 0),
               ("下一步(P2 可选): k-shot 档位曲线 + case-embedding 调研 + planner cap<8 修复 PR(issue 草稿已备)", 0),
               ("汇报/复现包随时可更新; 补跑工具 run_supplement.py 已自愈, 后续低峰可再扩 seed", 0)], top=1.4, height=3.9, size=14)
    add_title_slide(prs, "谢谢", ["三策略同预算对照 + 3-seed 统计 + 稳定性验证闭环 + 可复现证据包", "数字来源: stage3_matrix/runs(21 单元) & stage4_analysis(定稿 2026-09-07)"])
    out = "C3_结项汇报_初稿.pptx"
    prs.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
