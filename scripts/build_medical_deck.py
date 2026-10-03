"""Swap the Pets/Dogs pictures and results out of the lightning-talk deck and put
the HAM10000 / brain-tumor MRI ones in. Reads the original, writes a new file;
the source deck is never modified."""
from __future__ import annotations
import copy, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Emu, Inches, Pt

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
FIG = Path("/Users/ssuresh/gambit/results/medical_presentation/talk_figures/dropin")

prs = Presentation(str(SRC))
slides = list(prs.slides)


def shape_by_id(slide, sid):
    for sh in slide.shapes:
        if sh.shape_id == sid:
            return sh
    raise KeyError(f"no shape {sid}")


def replace_picture(slide, sid, path: Path):
    """Drop a new image into an old picture's box, preserving z-order and centre."""
    from PIL import Image
    old = shape_by_id(slide, sid)
    L, T, W, H = old.left, old.top, old.width, old.height
    iw, ih = Image.open(path).size
    box_ar, img_ar = W / H, iw / ih
    if img_ar >= box_ar:                      # wider than the box: fit width
        w, h = W, int(round(W / img_ar))
    else:                                     # taller: fit height
        w, h = int(round(H * img_ar)), H
    left, top = L + (W - w) // 2, T + (H - h) // 2
    new = slide.shapes.add_picture(str(path), left, top, w, h)
    old._element.addprevious(new._element)    # take the old shape's z-position
    old._element.getparent().remove(old._element)
    return new


def set_lines(tf, lines, sizes=None):
    """Rewrite a text frame line by line, keeping each paragraph's run formatting."""
    paras = tf.paragraphs
    for i, line in enumerate(lines):
        if i < len(paras):
            para = paras[i]
        else:
            para = copy.deepcopy(paras[-1]._p)
            paras[-1]._p.addnext(para)
            para = tf.paragraphs[-1]
        runs = para.runs
        if not runs:
            para.add_run()
            runs = para.runs
        runs[0].text = line
        for extra in runs[1:]:
            extra._r.getparent().remove(extra._r)
        if sizes and sizes[i] is not None:
            runs[0].font.size = Pt(sizes[i])
    for para in list(tf.paragraphs)[len(lines):]:
        para._p.getparent().remove(para._p)


def clone_last_row(tbl):
    """Append a row by deep-copying the last one, so it inherits its formatting."""
    last = tbl.rows[len(tbl.rows) - 1]._tr
    last.addnext(copy.deepcopy(last))


def clone_last_col(tbl):
    """Append a column by deep-copying the last one, keeping cell formatting."""
    gridcol = tbl.columns[len(tbl.columns) - 1]._gridCol
    gridcol.addnext(copy.deepcopy(gridcol))
    for row in tbl.rows:
        tr = row._tr
        tr.tc_lst[-1].addnext(copy.deepcopy(tr.tc_lst[-1]))


def set_cell(cell, text, size_pt=None):
    """Write a cell. A "\n" in `text` becomes a second paragraph, so a two-line
    header can wrap where it reads best rather than wherever the box runs out.

    `size_pt` matters: the deck's table is 20pt at four columns of 2.86". Five columns
    leaves 2.19", where a 20pt "EfficientNetV2-S" needs ~2.4" and wraps to a third line,
    overflowing the row and pushing the table into the footnote below it."""
    lines = text.split("\n")
    cell.margin_left = cell.margin_right = Inches(0.04)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    tf = cell.text_frame
    first = tf.paragraphs[0]
    for extra in list(tf.paragraphs)[1:]:
        extra._p.getparent().remove(extra._p)
    for i, line in enumerate(lines):
        if i == 0:
            para = first
        else:
            first._p.addnext(copy.deepcopy(first._p))
            para = tf.paragraphs[i]
        runs = para.runs
        if not runs:
            para.add_run()
            runs = para.runs
        runs[0].text = line
        if size_pt is not None:
            runs[0].font.size = Pt(size_pt)
        for r in runs[1:]:
            r._r.getparent().remove(r._r)


# Slide 1 and slide 3 are dataset-independent and are left exactly as they are.

# ---------------------------------------------------------------- slide 2
replace_picture(slides[1], 17, FIG / "slide2_problem_ham10000.png")
set_lines(shape_by_id(slides[1], 3).text_frame, [
    "What current methods give us",
    "Grad-CAM and Integrated Gradients build each class's map on its own, never "
    "looking at the other. The melanoma map and the nevus map land on the same "
    "pixels — cosine 0.82 on average across the validation set.",
])
set_lines(shape_by_id(slides[1], 18).text_frame,
          ["“Why melanoma rather than a benign nevus?”"])

# ---------------------------------------------------------------- slide 4
replace_picture(slides[3], 2, FIG / "slide4_method_ham10000.png")
set_lines(shape_by_id(slides[3], 6).text_frame,
          ["Keep the tiles a class was given, blur everything else, and re-run the very "
           "same frozen classifier. Every number in this talk comes off that one masked "
           "forward pass — no retraining, no annotations."])
set_lines(shape_by_id(slides[3], 7).text_frame,
          ["Two tiles out of 49, and they separate melanoma from nevus better than the "
           "whole image does: +0.39 against +0.07."])

# ---------------------------------------------------------------- slide 6
replace_picture(slides[5], 15, FIG / "slide6_hero_ham10000.png")
set_lines(shape_by_id(slides[5], 10).text_frame,
          ["Raw Grad-CAM (middle) covers the whole lesion for both diagnoses — cosine "
           "0.99 here. After allocation (right) the melanoma evidence holds the dark "
           "irregular core while the nevus evidence moves to the smooth lower border."])

# ---------------------------------------------------------------- slide 7
set_lines(shape_by_id(slides[6], 6).text_frame,
          ["Two datasets × two fine-tuned backbones  ·  Grad-CAM and Integrated "
           "Gradients as base evidence  ·  7×7 tiles, top-5 hypotheses"])
tbl_shape = shape_by_id(slides[6], 7)
tbl = tbl_shape.table
tbl_w, tbl_h = tbl_shape.width, tbl_shape.height   # cloning widens the frame; hold both
rows = [
    ["", "HAM10000\nResNet-18", "HAM10000\nEfficientNetV2-S",
     "Brain MRI\nResNet-18", "Brain MRI\nEfficientNetV2-S"],
    ["Overlap ↓", "0.447 → 0.010", "0.442 → 0.016", "0.067 → 0.002", "0.064 → 0.008"],
    ["Sufficiency ↑", "0.672 → 0.992", "0.268 → 1.491", "−0.001 → 1.508", "0.212 → 1.544"],
    ["Contrastive margin ↑", "−1.65 → −1.25", "−3.18 → −1.34", "−0.93 → +1.50", "−0.61 → +1.46"],
]
HEADER_PT, BODY_PT = 15, 16
while len(tbl.columns) < len(rows[0]):
    clone_last_col(tbl)
while len(tbl.rows) < len(rows):
    clone_last_row(tbl)
# The metric names are the longest strings in the table, so the label column gets the
# slack and the four dataset columns split what is left evenly.
label_w = int(Inches(2.70))
data_w = (tbl_w - label_w) // 4
tbl.columns[0].width = label_w
for c in range(1, 5):
    tbl.columns[c].width = data_w
row_h = int(tbl_h / len(rows))          # 4 rows over 3.77" is the deck's original 0.94"
for row in tbl.rows:
    row.height = row_h
for r, line in enumerate(rows):
    for c, txt in enumerate(line):
        set_cell(tbl.rows[r].cells[c], txt, HEADER_PT if r == 0 else BODY_PT)
tbl_shape.width, tbl_shape.height = tbl_w, tbl_h

set_lines(shape_by_id(slides[6], 11).text_frame,
          ["*Grad-CAM, n = 400. Overlap falls 98 / 96 / 97 / 88%. Integrated Gradients "
           "agrees: 0.495 → 0.030, 0.543 → 0.029, 0.326 → 0.010, 0.328 → 0.014. Mask "
           "budget is held only on HAM10000 / ResNet-18 (×1.02); the other three spend "
           "×1.12, ×1.38 and ×1.21, so part of their sufficiency gain is bought with "
           "extra highlight rather than relocated. With the shared mask on — the "
           "configuration the figures use — HAM10000 / ResNet-18 reads 0.447 → 0.069, "
           "holding at 0.050 ± 0.007 over three seeds. Sufficiency is a raw logit."])

# ---------------------------------------------------------------- slide 8
set_lines(shape_by_id(slides[7], 2).text_frame, [
    "Limitations",
    "The explanation is still bounded by the base evidence. If Grad-CAM is wrong, "
    "CDEA-Contrastive reallocates a wrong field.",
    "Lesion outlines cannot validate a class-unique mask — every HAM10000 class is a "
    "lesion, so the annotation is ground truth for the shared part. On dermoscopy a "
    "fixed centred rectangle beats every method measured.",
    "",
    "Summary",
    "Explanations for competing diagnoses should be allocated jointly, not computed "
    "one class at a time.",
    "CDEA-Contrastive decomposes the evidence into unique and shared regions by "
    "treating the candidate diagnoses as players competing for image tiles.",
    "Masks are verified by intervention, not asserted — removing a class's unique "
    "evidence costs that class and helps its rivals, where an equal-budget random "
    "deletion does nothing. The classifier is never retrained.",
])

DST.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(DST))
print("wrote", DST)
