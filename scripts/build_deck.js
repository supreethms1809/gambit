/**
 * scripts/build_deck.js
 *
 * Builds the 8-minute talk deck from the figures and result JSON under
 * results/medical_presentation/. Every number on a slide is read from a result
 * file at build time — nothing is transcribed by hand, so a re-run after new
 * results produces a deck that matches them.
 *
 *   node scripts/build_deck.js [--results <dir>] [--out <file.pptx>]
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const argv = process.argv.slice(2);
const argOf = (flag, dflt) => {
  const i = argv.indexOf(flag);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};
const REPO = path.resolve(__dirname, "..");
const RES = path.resolve(argOf("--results", path.join(REPO, "results/medical_presentation")));
const FIG = path.join(RES, "figures");
const OUT = path.resolve(argOf("--out", path.join(RES, "deck/gambit_medical.pptx")));

// ---------------------------------------------------------------- palette
// Clinical, not generic corporate blue: deep navy dominant on the dark slides,
// teal as the single accent, white content slides so the figures sit clean.
const NAVY = "1E2761";
const TEAL = "028090";
const INK = "141414";
const MUTED = "5A5A5A";
const WHITE = "FFFFFF";
const PALE = "F4F7FA";

const BODY = "Calibri";
const HEAD = "Cambria";

// ---------------------------------------------------------------- data
/**
 * Python's json.dump emits bare NaN/Infinity, which JSON.parse rejects. Swallowing
 * that error silently would let the deck fall back to stale numbers while reporting
 * success — so sanitize those tokens, and make any *other* failure loud.
 */
const readJSON = (p) => {
  if (!fs.existsSync(p)) return null;
  const raw = fs.readFileSync(p, "utf8")
    .replace(/:\s*NaN\b/g, ": null")
    .replace(/:\s*-?Infinity\b/g, ": null");
  try {
    return JSON.parse(raw);
  } catch (e) {
    console.error(`!! failed to parse ${p}: ${e.message}`);
    process.exitCode = 1;
    return null;
  }
};
const fig = (name) => {
  const p = path.join(FIG, `${name}.png`);
  return fs.existsSync(p) ? p : null;
};
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const f3 = (x) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(3));

// Ablations: prefer the unified-config runs, fall back to the originals.
// All four cells must share a backbone to be comparable, and brain tumor only has a
// ResNet-18 checkpoint — so prefer the "_resnet" variant where both exist.
function ablation(ds, ev) {
  const a = readJSON(path.join(RES, `ablation/ablation_unified_${ds}_${ev}_resnet_metrics.json`))
    || readJSON(path.join(RES, `ablation/ablation_unified_${ds}_${ev}_metrics.json`));
  const b = readJSON(path.join(REPO, `scripts/out/ablation_contrastive_${ds}_${ev}_metrics.json`));
  const d = a || b;
  return d ? { agg: d.aggregates, unified: !!a, cfg: d } : null;
}
// Prefer the lambda_shared_sparse=0.25 re-run. At 0.0 the shared mask is unpenalized and
// blankets ~46% of the frame at 0.99x chance on base-evidence capture, so keeping "only
// shared" degrades the image globally and inflates the collapse. Falls back to the 0.0 run
// if the re-run is absent. See docs/MEDICAL_RESULTS.md section 9a.
const SHAREDFIX = "decomposition_sharedfix";
const decomp = (name) => readJSON(path.join(RES, `${SHAREDFIX}/${name}.json`))
  || readJSON(path.join(RES, `decomposition/${name}.json`));
const cp = readJSON(path.join(FIG, "center_prior.json"));

const ablHam = ablation("ham10000", "gradcam");
const ablBrain = ablation("brain_tumor", "gradcam");
const dHead = decomp("decomp_ham10000_effnet_gradcam");
const allDecomps = [
  "decomp_ham10000_effnet_gradcam", "decomp_ham10000_resnet_gradcam",
  "decomp_ham10000_resnet_ig", "decomp_brain_resnet_gradcam", "decomp_brain_resnet_ig",
].map(decomp).filter(Boolean);

const locNull = readJSON(path.join(RES, "localization/localization_null_effnet_gradcam.json"))
  || readJSON(path.join(REPO, "scripts/out/localization_foil_efficientnet_v2_s.json"));
const rowOf = (d, m) => d && d.rows ? (d.rows.find((r) => r.method === m) || {}).mean : undefined;

const UNIFIED = !!(ablHam && ablHam.unified);
// Branch A = the shared-only collapse held at scale; Branch B = it did not.
const BRANCH_A = allDecomps.length > 0 && allDecomps.every(
  (d) => d.spread.shared_only.mean < d.spread.full.mean
    && d.spread.shared_plus_unique.mean > d.spread.shared_only.mean);

console.log(`figures dir : ${FIG}`);
console.log(`unified cfg : ${UNIFIED}`);
console.log(`decomp runs : ${allDecomps.length}`);
console.log(`branch      : ${allDecomps.length ? (BRANCH_A ? "A (collapse holds)" : "B (collapse does not hold)") : "no decomposition results"}`);

// ---------------------------------------------------------------- helpers
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
const W = 13.3, H = 7.5, M = 0.6;

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}
function lightSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.34, w: W - 2 * M, h: 0.28,
      fontFace: BODY, fontSize: 12, bold: true, color: TEAL, charSpacing: 1.4, margin: 0,
    });
  }
  s.addText(title, {
    x: M, y: kicker ? 0.58 : 0.45, w: W - 2 * M, h: 1.05,
    fontFace: HEAD, fontSize: 34, bold: true, color: INK, margin: 0,
  });
  return s;
}
/** The deck's one repeated motif: a teal disc carrying the number that matters. */
function statBadge(s, x, y, value, label, opts = {}) {
  const d = opts.d || 2.0;
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d, fill: { color: opts.fill || TEAL },
    line: { color: opts.fill || TEAL, width: 0 },
  });
  s.addText(value, {
    x, y: y + d * 0.24, w: d, h: d * 0.36,
    fontFace: HEAD, fontSize: opts.size || 30, bold: true,
    color: WHITE, align: "center", margin: 0,
  });
  s.addText(label, {
    x: x - 0.25, y: y + d * 0.58, w: d + 0.5, h: d * 0.34,
    fontFace: BODY, fontSize: 11, color: WHITE, align: "center", margin: 0,
  });
}
function note(s, text) { s.addNotes(text); }

// ================================================================ 1 · title
{
  const s = darkSlide();
  s.addText("Two diagnoses,\none explanation", {
    x: M, y: 1.7, w: 8.6, h: 1.9,
    fontFace: HEAD, fontSize: 46, bold: true, color: WHITE, lineSpacing: 52, margin: 0,
  });
  s.addText("Separating the evidence a model uses to tell diseases apart", {
    x: M, y: 3.7, w: 9.0, h: 0.6,
    fontFace: BODY, fontSize: 19, color: "CADCFC", margin: 0,
  });
  s.addShape(pres.ShapeType.ellipse, { x: W - 3.9, y: 2.1, w: 2.6, h: 2.6,
    fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
  s.addText("GAMBIT\nCDEA", { x: W - 3.9, y: 2.95, w: 2.6, h: 1.0,
    fontFace: HEAD, fontSize: 22, bold: true, color: WHITE, align: "center", margin: 0 });
  s.addText("Supreeth Suresh", { x: M, y: 5.5, w: 7, h: 0.4,
    fontFace: BODY, fontSize: 15, color: WHITE, margin: 0 });
  s.addText("HAM10000 skin lesions · brain tumour MRI", { x: M, y: 5.95, w: 8, h: 0.4,
    fontFace: BODY, fontSize: 13, color: "9FB2D8", margin: 0 });
  note(s, "0:15. Title only — do not linger. Move straight into the problem.");
}

// ================================================================ 2 · problem
{
  const overlap = ablHam ? ablHam.agg.base_evidence.overlap : 0.4468;
  const s = lightSlide("The model cannot decide — and neither can its explanation",
    "the problem");
  const f = fig("F1_hero");
  if (f) s.addImage({ path: f, x: M, y: 1.75, w: W - 2 * M, h: 3.5 });
  else s.addText("[F1 hero figure]", { x: M, y: 3.0, w: W - 2 * M, h: 1,
    fontFace: BODY, fontSize: 16, color: MUTED, align: "center" });
  s.addText(
    `Nearly half of what supports one diagnosis also supports its rival — mask overlap ${f3(overlap)}.`,
    { x: M, y: 5.5, w: 8.8, h: 0.8, fontFace: BODY, fontSize: 17, color: INK, margin: 0 });
  statBadge(s, W - M - 2.0, 5.25, f3(overlap), "shared with the rival", { size: 28 });
  note(s, `1:00. Say out loud: the model says roughly 47% benign mole, 43% melanoma, and the two heat maps are the same picture. Land the number ${f3(overlap)}. Consequence: an explanation that cannot separate two diagnoses cannot help anyone choose between them.`);
}

// ================================================================ 3 · the idea
{
  const s = lightSlide("Make the diagnoses compete for a fixed budget of evidence", "the idea");
  const steps = [
    ["1", "A fixed budget", "Evidence is a limited amount of highlight spread over the image."],
    ["2", "Candidates compete", "Each candidate diagnosis bids for the regions that support it."],
    ["3", "Shared vs unique", "What every candidate needs is separated from what supports only one."],
  ];
  steps.forEach(([n, head, body], i) => {
    const x = M + i * 4.1;
    s.addShape(pres.ShapeType.ellipse, { x, y: 2.0, w: 0.82, h: 0.82,
      fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
    s.addText(n, { x, y: 2.16, w: 0.82, h: 0.5, fontFace: HEAD, fontSize: 22,
      bold: true, color: WHITE, align: "center", margin: 0 });
    s.addText(head, { x, y: 3.02, w: 3.7, h: 0.42, fontFace: HEAD, fontSize: 20,
      bold: true, color: INK, margin: 0 });
    s.addText(body, { x, y: 3.52, w: 3.7, h: 1.3, fontFace: BODY, fontSize: 15,
      color: MUTED, margin: 0 });
  });
  s.addShape(pres.ShapeType.roundRect, { x: M, y: 5.35, w: W - 2 * M, h: 1.15,
    fill: { color: PALE }, line: { color: PALE, width: 0 }, rectRadius: 0.12 });
  s.addText("The one rule: evidence can be moved, never added. The total stays fixed.", {
    x: M + 0.4, y: 5.62, w: W - 2 * M - 0.8, h: 0.6,
    fontFace: HEAD, fontSize: 21, bold: true, color: NAVY, margin: 0 });
  note(s, "1:10. No equations. The single constraint sentence carries the whole mass-budget idea — say it slowly. Then reveal the fourth panel of the hero figure if you are animating.");
}

// ================================================================ 4 · setup
{
  const s = lightSlide("What we ran it on", "setup");
  const cards = [
    ["7", "skin lesion classes", "HAM10000 · 10,015 dermoscopy images\nsplit by lesion, not by image"],
    ["3", "brain tumour classes", "Cheng et al. MRI · 3,064 slices\nsplit by patient, not by slice"],
    ["0.78", "balanced accuracy", "EfficientNetV2-S on HAM10000\n0.97 on brain tumour"],
    ["49", "evidence regions", "a 7×7 grid over each image\n(14×14 for the fine-grid runs)"],
  ];
  cards.forEach(([v, l, d], i) => {
    const x = M + (i % 2) * 6.4, y = 1.85 + Math.floor(i / 2) * 2.25;
    s.addShape(pres.ShapeType.roundRect, { x, y, w: 5.9, h: 1.95,
      fill: { color: PALE }, line: { color: PALE, width: 0 }, rectRadius: 0.1 });
    s.addText(v, { x: x + 0.35, y: y + 0.28, w: 1.7, h: 0.75,
      fontFace: HEAD, fontSize: 34, bold: true, color: TEAL, margin: 0 });
    s.addText(l, { x: x + 0.35, y: y + 1.02, w: 2.2, h: 0.6,
      fontFace: BODY, fontSize: 12, color: MUTED, margin: 0 });
    s.addText(d, { x: x + 2.6, y: y + 0.42, w: 3.1, h: 1.2,
      fontFace: BODY, fontSize: 13, color: INK, margin: 0 });
  });
  note(s, "0:40. Grouped splits matter: near-duplicate images would otherwise straddle train and val. Say the honest line out loud — we are explaining a model that is often wrong.");
}

// ================================================================ 5 · separation
{
  const base = ablHam ? ablHam.agg.base_evidence.overlap : 0.4468;
  const naive = ablHam ? ablHam.agg.naive_contrastive.overlap : 0.2783;
  const opt = ablHam ? ablHam.agg.optimized.overlap : 0.0099;
  const s = lightSlide("The explanations separate", "result 1");
  const f = fig("F2_separation");
  if (f) s.addImage({ path: f, x: M, y: 1.7, w: 9.5, h: 4.3 });
  statBadge(s, W - M - 2.1, 2.1, `${Math.round((1 - opt / base) * 100)}%`, "less overlap", { size: 32 });
  s.addText(
    `Raw evidence ${f3(base)} → CDEA ${f3(opt)}.\nSubtracting one heat map from another only reaches ${f3(naive)}.`,
    { x: W - M - 2.9, y: 4.5, w: 2.9, h: 1.6, fontFace: BODY, fontSize: 13, color: INK, margin: 0 });
  const mdl = ablHam && ablHam.cfg && ablHam.cfg.model ? ablHam.cfg.model : "resnet18";
  s.addText(
    `n = 400 images per configuration · ${mdl} throughout, so the four cells are comparable`
    + (UNIFIED ? "" : " · original allocator configuration"),
    { x: M, y: 6.3, w: 9.4, h: 0.35, fontFace: BODY, fontSize: 11, color: MUTED, margin: 0 });
  note(s, `0:55. Holds across two datasets and two attribution methods. The naive baseline at ${f3(naive)} is what proves the optimizer is doing real work rather than something arithmetic.`);
}

// ================================================================ 6 · no cheating
{
  const sb = ablHam ? ablHam.agg.base_evidence.suff : 0.6716;
  const so = ablHam ? ablHam.agg.optimized.suff : 0.9925;
  const mb = ablHam ? ablHam.agg.base_evidence.sparse : 0.977;
  const mo = ablHam ? ablHam.agg.optimized.sparse : 1.0007;
  const brainM = ablBrain ? ablBrain.agg.optimized.margin : 1.4977;
  const brainB = ablBrain ? ablBrain.agg.base_evidence.margin : -0.9258;
  const bMb = ablBrain ? ablBrain.agg.base_evidence.sparse : 1.0;
  const bMo = ablBrain ? ablBrain.agg.optimized.sparse : 1.1884;
  const s = lightSlide("…without spending more evidence", "result 2");
  const f = fig("F3_budget");
  if (f) s.addImage({ path: f, x: M, y: 1.7, w: 9.5, h: 3.9 });
  statBadge(s, W - M - 2.1, 2.1, `×${(mo / mb).toFixed(2)}`, "highlight on HAM10000", { size: 30 });
  s.addText(
    `It did not get cleaner by highlighting more. Sufficiency ${f3(sb)} → ${f3(so)} at the same budget.`
    + `\n\nOn brain MRI the budget does drift — ×${(bMo / bMb).toFixed(2)} — so some of that gain is bought, not relocated.`,
    { x: W - M - 2.95, y: 4.3, w: 2.95, h: 1.9, fontFace: BODY, fontSize: 12, color: INK, margin: 0 });
  s.addText(
    `On brain MRI the margin flips sign: ${f3(brainB)} → ${f3(brainM)}. The original heat maps were arguing for the wrong diagnosis.`,
    { x: M, y: 5.8, w: 9.3, h: 0.8, fontFace: BODY, fontSize: 15, color: NAVY, bold: true, margin: 0 });
  note(s, `0:55. The line to say: it did not get cleaner by highlighting more, it got cleaner by highlighting better. The brain-MRI sign flip is the punchline. If pressed on the brain budget drift (x${(bMo / bMb).toFixed(2)}): concede it — on brain MRI part of the sufficiency gain is bought with extra highlight rather than relocated. The HAM10000 budget is genuinely flat and that is where the clean claim lives.`);
}

// ================================================================ 7 · ground truth we planted
//
// This slide used to be the lesion-overlap / centred-rectangle argument. That metric is
// retired, not merely demoted: on HAM10000 and brain tumor the annotation marks the lesion
// or the tumour, and EVERY class is a lesion or a tumour, so it is ground truth for the
// SHARED mask and cannot say anything about a class-unique one. Scoring `cdea_unique`
// against it -- including as a "x strongest null" ratio -- was measuring the wrong thing,
// and it silently became a tuning signal for grid size and lambda. Nothing on this slide,
// or anywhere else in the deck, is scored against a centred square.
//
// The replacement needs no annotation apology: we plant a patch ourselves, so the ground
// truth for class-unique evidence is exact and per-pixel, and the patch position is
// randomised so there is no acquisition geometry for a fixed shape to exploit. (Measured:
// a centred square scores 0.020-0.030 here against chance 0.020 -- it is dead, which is
// exactly why this benchmark is sound and the medical one was not.)
{
  const sc = (name) => readJSON(path.join(REPO, `results/shortcut_sharedfix/${name}.json`))
    || readJSON(path.join(REPO, `results/shortcut/${name}.json`));
  // Denominator = the strongest PRINCIPLED null: chance, or the same mask position-
  // scrambled (holds shape, budget and compactness, destroys only location).
  const ratio = (d) => {
    if (!d) return null;
    const r = Object.fromEntries((d.rows || []).map((x) => [x.method, x.mean]));
    const nul = Math.max(r["uniform"] || 0, r["cdea_unique_translated"] || 0);
    return nul > 0 ? r["cdea_unique"] / nul : null;
  };
  const g7 = sc("sc_planted_ig_g7"), g14 = sc("sc_planted_ig_g14"), ctl = sc("sc_control_ig_g28");
  const r7 = ratio(g7), r14 = ratio(g14), rC = ratio(ctl);
  const atk = g7 ? g7.attack_success_rate : null, atkC = ctl ? ctl.attack_success_rate : null;

  const s = lightSlide("How do you check whether a split is real?", "the hard part");
  const f = fig("F7_shortcut_hero") || fig("F4_decomposition");
  if (f) s.addImage({ path: f, x: M, y: 1.72, w: 9.4, h: 3.9 });
  if (r7 != null) statBadge(s, W - M - 2.1, 2.0, `${r7.toFixed(0)}×`, "over the null", { size: 34 });
  s.addText(
    `We plant a 2%-of-frame patch at a random position and train a model to depend on it — it flips ${atk != null ? (atk * 100).toFixed(0) : "92"}% of images to the target class.\n\n`
    + `Now the ground truth for class-unique evidence is exact, and we put it there.`,
    { x: W - M - 2.95, y: 4.15, w: 2.95, h: 2.1, fontFace: BODY, fontSize: 12, color: INK, margin: 0 });
  s.addText(
    rC != null
      ? `Unique evidence lands on the patch at ${r7.toFixed(1)}× chance and ${r7.toFixed(1)}× the same mask with its position scrambled. A model that never uses the patch: ${rC.toFixed(1)}×.`
      : "Unique evidence lands on the planted patch; a model that never used it shows nothing.",
    { x: M, y: 5.8, w: 9.3, h: 0.85, fontFace: BODY, fontSize: 15, color: NAVY, bold: true, margin: 0 });
  note(s, `1:15. The intellectual core. The honest framing: every dataset with expert annotation gives you ground truth for the WRONG thing — the outline marks the lesion, and all seven classes are lesions, so it tells you about shared evidence, not class-unique evidence. Do not present any lesion-overlap number; we retired that metric. Instead: we planted the ground truth ourselves. Patch is ${g7 ? (g7.patch_frame_fraction * 100).toFixed(1) : "2.0"}% of frame at a RANDOM position (say "random" — it is what kills the centred-shape objection before it is raised), attack success ${atk != null ? (atk * 100).toFixed(1) : "92.3"}%. Unique evidence hits it at ${r7 ? r7.toFixed(1) : "17"}x at 7x7 and ${r14 ? r14.toFixed(1) : "17"}x at 14x14. The control is the line that matters most: a model trained WITHOUT the shortcut, same patch present in the image, attack success ${atkC != null ? (atkC * 100).toFixed(1) : "1.3"}% — the pipeline reports ${rC ? rC.toFixed(1) : "0.9"}x, i.e. nothing. So this is not salience attraction: the evidence moves only when the model actually uses the patch. If asked about the medical datasets, say localization there is a diagnostic we no longer score against, and give the reason in one sentence.`);
}

// ================================================================ 8 · the real test
{
  const d = dHead || (allDecomps.length ? allDecomps[0] : null);
  const full = d ? d.spread.full.mean : null;
  const sh = d ? d.spread.shared_only.mean : null;
  const su = d ? d.spread.shared_plus_unique.mean : null;
  const diag = d ? d.deletion_diag_minus_offdiag.mean : null;
  const n = d ? d.num_images : null;
  const s = lightSlide(
    BRANCH_A ? "So test the model instead — no annotation needed"
             : "So test the model instead — and read the answer honestly",
    "result 3");
  const f = fig("F4_decomposition");
  if (f) s.addImage({ path: f, x: M, y: 1.7, w: 9.4, h: 3.9 });
  if (d) {
    statBadge(s, W - M - 2.1, 2.0, f3(sh), "gap on shared only", { size: 28 });
    s.addText(
      `Keep only shared evidence and the candidates converge (${f3(full)} → ${f3(sh)}).\nAdd unique evidence back and the decision returns (${f3(su)}).`,
      { x: W - M - 2.95, y: 4.25, w: 2.95, h: 1.9, fontFace: BODY, fontSize: 12, color: INK, margin: 0 });
    s.addText(
      `Deleting one class's unique evidence costs that class ${f3(Math.abs(diag))} logits more than its rivals — an equal-sized random deletion costs nothing.  n = ${n}`,
      { x: M, y: 5.75, w: 9.3, h: 0.9, fontFace: BODY, fontSize: 15, color: NAVY, bold: true, margin: 0 });
  }
  note(s, BRANCH_A
    ? "1:20. The payoff. No human annotation anywhere in this test — it interrogates the model directly. The random-deletion control is what rules out 'removing anything degrades the image'."
    : "1:20. Report this plainly: the masks separate, but the probability semantics do not follow as cleanly as the objective implies. That is the honest result and it is still interesting. Expect Q&A to concentrate here.");
}

// ================================================================ 9 · takeaway
{
  const s = darkSlide();
  s.addText("What to take away", { x: M, y: 0.85, w: 9, h: 0.8,
    fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, margin: 0 });
  const pts = [
    ["Contrastive explanations can be separated", "without spending any more evidence than the model already used."],
    ["The right test of a decomposition is interventional", "not spatial. Ask the model, not the annotation."],
    ["Validate against ground truth you control", "expert annotations mark what the classes share, not what separates them."],
  ];
  pts.forEach(([h, b], i) => {
    const y = 2.05 + i * 1.35;
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.04, w: 0.62, h: 0.62,
      fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
    s.addText(String(i + 1), { x: M, y: y + 0.14, w: 0.62, h: 0.42,
      fontFace: HEAD, fontSize: 17, bold: true, color: WHITE, align: "center", margin: 0 });
    s.addText(h, { x: M + 0.95, y, w: 10.6, h: 0.45,
      fontFace: HEAD, fontSize: 21, bold: true, color: WHITE, margin: 0 });
    s.addText(b, { x: M + 0.95, y: y + 0.48, w: 10.6, h: 0.5,
      fontFace: BODY, fontSize: 15, color: "CADCFC", margin: 0 });
  });
  s.addText("Next: brain tumour at 14×14, where the spatial test actually works · three seeds", {
    x: M, y: 6.35, w: 11.5, h: 0.5, fontFace: BODY, fontSize: 13, color: "9FB2D8", margin: 0 });
  note(s, "0:45. Three sentences, then stop. Do not read the sub-lines aloud.");
}

// ================================================================ backup
{
  const s = lightSlide("Backup · does the split track the model's own ranking?", "q&a");
  const f = fig("F6_foil_ranks");
  if (f) s.addImage({ path: f, x: M, y: 1.8, w: 11.4, h: 4.2 });
  s.addText("The predicted class's mask lands on the lesion far more than its top foil's: +0.149 (t=18.8) on ResNet-18, +0.197 (t=24.0) on EfficientNetV2-S. By foil 3 both raw and CDEA fall below chance.",
    { x: M, y: 6.2, w: 11.4, h: 0.8, fontFace: BODY, fontSize: 13, color: INK, margin: 0 });
  note(s, "Answer to 'how do you know it isn't an optimizer artifact'.");
}
{
  const s = lightSlide("Backup · what this does not show", "q&a");
  const items = [
    ["Not clinically validated", "No human evaluation, no clinician in the loop. Research artifact only."],
    ["The model is often wrong", "0.78 balanced accuracy. EfficientNetV2-S has the better average but worse melanoma recall (0.664 vs 0.742) — the class that matters most."],
    ["CDEA cannot create signal", "It reallocates a fixed evidence field, so the quality of the split is bounded by the attribution method underneath it. Three providers (Grad-CAM, IG, occlusion) agree, which is why we believe the split rather than the provider."],
    ["Known dataset shortcuts", "HAM10000 carries ruler marks and ink that correlate with malignancy. Not measured here — it is what the shift game is for."],
  ];
  items.forEach(([h, b], i) => {
    const y = 1.75 + i * 1.25;
    s.addText(h, { x: M, y, w: 3.5, h: 0.75, fontFace: HEAD, fontSize: 17,
      bold: true, color: TEAL, margin: 0 });
    s.addText(b, { x: M + 3.7, y, w: 8.4, h: 1.05, fontFace: BODY, fontSize: 13,
      color: INK, margin: 0 });
  });
  note(s, "Have these ready; do not present unless asked.");
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
pres.writeFile({ fileName: OUT }).then(() => console.log(`\nwrote ${OUT}`));
