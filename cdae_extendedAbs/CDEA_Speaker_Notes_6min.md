# CDEA-Contrastive — Speaker Notes

**ICAIIS 2026 · Paper 101 · London, 16–18 Aug 2026**
Deck: `CDEA_Contrastive_6min_v10.pptx` · 9 main slides + 3 backup · target **6:00**

These notes are also embedded in the .pptx (Presenter View shows them per slide). This
file is the rehearsal copy.

**Rule for this talk:** every technical term gets a plain-English gloss the first time it
appears. If a sentence needs a definition the audience doesn't have yet, cut it.

---

## Timing at a glance

| # | Slide | Window | Length | Cumulative |
| --- | --- | --- | --- | --- |
| 1 | Title | 0:00–0:15 | 15 s | 0:15 |
| 2 | The problem | 0:15–1:00 | 45 s | 1:00 |
| 3 | The idea | 1:00–1:45 | 45 s | 1:45 |
| 4 | The method — one score | 1:45–2:45 | 60 s | 2:45 |
| 5 | Checked by covering up | 2:45–3:35 | 50 s | 3:35 |
| 6 | Results | 3:35–4:45 | 70 s | 4:45 |
| 7 | Worked example | 4:45–5:15 | 30 s | 5:15 |
| 8 | Honest assessment | 5:15–5:45 | 30 s | 5:45 |
| 9 | Takeaway | 5:45–6:00 | 15 s | **6:00** |

Slides 4 and 6 are the two long ones. If you are running late at the 3:35 mark, the
recoverable time is in slide 6 — drop the Stanford Dogs sentence, not the margin result.

---

## Slide 1 — Title · 0:00–0:15

> Good morning. I'm Supreeth Suresh from the University of Wyoming, with my advisor
> Suresh Muknahallipatna.
>
> In six minutes I want to convince you of one idea: when several classes compete for a
> prediction, their explanations should be worked out together, not one at a time.

**Delivery:** say the title, then move on. Do not read the affiliation off the slide.

## Slide 2 — The problem · 0:15–1:00 (~45 s)

> Here is the problem, and it is easy to state.
>
> Today's explanation tools draw you a heat-map: which parts of the picture did the model
> use. The catch is that you can draw one for **any** class. So you draw one for cat, then
> you draw one for dog — and they come out looking nearly the same. Both light up the animal.
>
> The fourth panel is the overlap: the pixels both classes claim at once. That region is
> doing no work in telling the two apart, and nothing in either map tells you that.
>
> This is fine if your question is "why cat". It falls apart the moment you ask "why cat
> **rather than** dog" — which is the question people actually ask.

**Point at:** the "claimed by BOTH" panel.

## Slide 3 — The idea · 1:00–1:45 (~45 s)

> So here is the reframing. Stop treating the top few classes as separate questions. Treat
> them as rivals competing over one image.
>
> Think of three doctors looking at the same X-ray. All three circle the same shadow. That
> agreement tells you nothing about which one is right. What you want are the smaller
> places where they **disagree**.
>
> Four steps. One: start from a heat-map you already trust, coarsened to a 7-by-7 grid of
> tiles. Two: split it into the part only this class claims, plus an optional shared part.
> Three: test any split by covering up — keep a class's tiles, blur the rest, re-run the
> same frozen network. Four: tune the masks against a single score.
>
> The split is the whole point. The unique part is the discriminative evidence. The shared
> part absorbs the common ground so it cannot pose as a reason to choose.

**Be honest:** the shared mask is optional and is switched **off** in these runs. Say so if
it comes up.

## Slide 4 — The method · 1:45–2:45 (~60 s)

> This is the entire method on one line.
>
> The players are the top few predicted classes. The prize is the set of image tiles. This
> score is the rule that settles who gets what.
>
> Two things we want **more** of. Sufficiency: show the model only a class's own tiles, and
> it should still recognise that class. Margin: under those same tiles, the class has to
> beat its closest rival — that is the contrastive term, and it is the one that carries our
> result.
>
> Three things we **penalise**. Overlap — two classes may not claim the same tile, and that
> is what actually makes them compete. Size, so the mask stays readable. Drift, so we stay
> anchored to the original heat-map.
>
> One thing worth stressing: the method cannot **add** highlight. It has a fixed budget and
> can only move it around. So any improvement is a relocation, not extra ink.
>
> All of this is gradient descent on the mask values only. The classifier is frozen.

**Be honest if asked:** no Nash, no Shapley, no core. It is a penalised joint objective
described in game terms. That is on the open-problems slide.

## Slide 5 — Checked by covering up · 2:45–3:35 (~50 s)

> How do we know an allocation is any good? We don't take its word for it — we cover things
> up and see what happens.
>
> Take one class's mask, keep those tiles, blur everything else, push that back through the
> frozen classifier. Everything we report comes off that one masked forward pass.
>
> In this example the masked image actually scores the cat **higher** than the original —
> plus 1.37 against plus 0.71. The tiles the allocator kept carry the class better than the
> whole picture does.
>
> Three numbers. Sufficiency: show only these tiles, does the model still recognise the
> class. Margin: do these same tiles favour this class over the runner-up. Overlap: are two
> classes still sitting on the same tiles.

**Be honest:** overlap and size are terms in our loss as well as numbers we report. That is
a circularity, and it is on the limitations slide.

## Slide 6 — Results · 3:35–4:45 (~70 s)

> Same three questions, now across four datasets. Fine-tuned ResNet-18, full test sets.
>
> **Overlap** first — do the two maps still point at the same pixels? It falls 78 to 99
> percent on MNIST and CIFAR-10, about 92 on Pets, about 97 on Stanford Dogs. But be honest
> about what that means: overlap is something we optimise, so this only tells you the
> optimiser worked. It is not proof of a better explanation.
>
> **Sufficiency** — does the class survive being shown only its own tiles? On MNIST and
> CIFAR-10, essentially unchanged, so we stripped out redundancy without breaking the
> prediction. On the binary Pets task it improves by up to 54 percent.
>
> **Margin** — and this is the actual result. On Pets it moves from roughly zero to 0.54.
> The raw attribution gave almost no separation between the two classes. The allocation
> produced some.
>
> On 120-class Stanford Dogs the class does not survive masking. Fine-grained recognition
> does not work yet — a coarse spatial mask cannot preserve texture-level detail. Overlap
> and margin still move the right way.

**If asked "percent of what":** sufficiency is a raw logit with an arbitrary zero, and it is
negative on three of our four datasets. That is why we only quote a percentage on Pets. The
change is +0.26.

**Cut first if late:** the Stanford Dogs sentence. Never cut the margin result.

## Slide 7 — Worked example · 4:45–5:15 (~30 s)

> One picture where the classifier is genuinely torn — 51 percent cat against 49 percent
> dog. So "why cat rather than dog" is a live question here, not a rhetorical one.
>
> The two middle columns are the raw Grad-CAM evidence. Both cover the animal. Useless for
> telling them apart.
>
> The two right columns are our allocation. The cat evidence stays on the kitten. The dog
> evidence moves off the animal entirely, onto the pink bedding underneath.
>
> That is the useful output: the near-tie is being carried by the background, not by the pet.

**Delivery:** 30 seconds. Point at the dog mask on the blanket, then move on.

## Slide 8 — Honest assessment · 5:15–5:45 (~30 s)

> Where this is still weak. This is preliminary work and I would rather name the gaps than
> have you find them.
>
> First, circularity. Overlap, sufficiency and margin are terms in our loss **and** the
> numbers on the results slide. We need a faithfulness test we do not optimise.
>
> Second, when we cover tiles up we fill them with blur, and blur is the only fill we ever
> tested.
>
> Third, we say the decomposition is unique but we have no proof and no seed study.
>
> Fourth, the fine-grained case still fails — and that is exactly where a contrastive
> explanation would be most valuable.
>
> Next steps are the matching list: real published baselines, a named solution concept, and
> evaluation from outside the objective.

**Delivery:** brisk, no apology. This slide buys credibility.

## Slide 9 — Takeaway · 5:45–6:00 (~15 s)

> To close. If several classes are competing, their explanations should be worked out
> together rather than one at a time.
>
> CDEA-Contrastive does that with an allocation that is checked by covering things up, on a
> classifier that is never retrained, on top of attribution methods you already use.
>
> The gains are real where the classes differ in an obvious way. Fine-grained recognition is
> still open.
>
> Thank you — happy to take questions.

**Stop talking here.** Do not add a summary sentence.

---

# Q&A preparation

## Backup slide 10 — "Why not just subtract the two maps?"

The single most likely question. Go to the slide.

> Difference-of-maps **wins** on overlap — it drives it to exactly zero, better than we do.
> And it gains nothing predictive: sufficiency stays at 0.464 against our 0.727, margin at
> 0.006 against our 0.537.
>
> That is the cleanest evidence that low overlap by itself is not the goal. Anyone can make
> two maps disjoint. Making them disjoint **and** individually sufficient under intervention
> is the hard part.

No CEM or counterfactual comparison yet — on the next-steps list.

## Backup slide 11 — Setup

**On seeds:** the seed sweep does not vary anything, because the checkpoint is reused across
seeds and the allocator initialises deterministically from the evidence map. The spread we
quote is across images, not seeds. Say that plainly rather than implying seed robustness we
have not measured.

## Backup slide 12 — A failure case

True dog, called a cat, 53 to 47. The allocation does not correct the mistake — it localises
the evidence behind each side of a near-tie, so you can see the cat claim rests on the face
and the human hand.

## Not on a slide — questions to have answers ready for

**"Have you tried this on anything that matters — medical images?"**

> Yes, since submission. We've extended to HAM10000 skin lesions — 10,015 dermoscopy images,
> 7 classes — with lesion-grouped splits so the same lesion never spans train and test. The
> useful thing there is that the dataset ships expert lesion outlines, so for the first time
> we can check whether the allocated masks land on the actual pathology instead of just
> eyeballing them. Early result: the unique masks land on the lesion substantially more than
> raw Grad-CAM does, and the shared mask sits at chance — which is what the decomposition
> predicts. That work is not in this abstract.

Keep it to that. **Do not quote localization numbers**: a fixed box at the image centre
scores higher than our masks on that metric, because dermoscopy images are centre-framed.
Until we have a position-controlled null, those numbers are not defensible in public.

**"Is this clinically validated?"** No. Research artifacts only. Say it flatly.

**"How expensive is it?"** 40 gradient steps on the mask values per image, classifier frozen
— no retraining, no architecture change. It sits on top of an attribution method you are
already running.

**"Why a 7×7 grid — isn't that coarse?"** Yes, and that is a real limit. It matches the final
convolutional feature map of ResNet-18 at 224 px. For targets smaller than one tile the
method cannot localise them at all. A finer grid — a ViT at 14×14 — is the obvious next step.

**"Does it work on transformers?"** Not tested at the time of this abstract. ResNet-18 only.

**"What if the top-m classes aren't the right hypotheses?"** Fair — m is a hyperparameter, 5
here and 2 on the binary task. Selecting hypotheses is out of scope; we take the model's own
top-m.

---

# Rehearsal checklist

- [ ] Full run-through under 6:00 without rushing slide 6
- [ ] Slide 2: know which panel is "claimed by BOTH" before you point
- [ ] Slide 4: say the fixed-budget line — it pre-empts "aren't you just adding highlight?"
- [ ] Slide 6: say "overlap is something we optimise" **unprompted**, don't wait to be caught
- [ ] Slide 8: brisk and unapologetic
- [ ] Slide 9: stop talking at "questions"
- [ ] Know the route to backup slide 10 without hunting
