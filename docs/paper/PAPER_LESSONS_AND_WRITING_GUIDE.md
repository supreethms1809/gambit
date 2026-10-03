# Lessons from a major revision, and a guide for writing the next paper

These notes come from a paper that went through review, received a major-revision decision, and was rebuilt over many drafts before resubmission. They are written to be general. Part 1 lists the mistakes and how to check for them. Part 2 describes how to write the paper so that a good version is reached in a few drafts rather than many. Part 3 gives a workflow, and Part 4 a checklist to run before submission.

---

## Part 1. Mistakes to avoid

Each item states the mistake, why it matters, and how to verify that the new paper does not repeat it.

### 1.1 Problem formulation

**Measuring the wrong target.** The method claimed to explain a model, but the evaluation measured agreement with the ground-truth labels. An explanation of a model must be judged against that model's outputs. Agreement with labels measures something else (data purity, or the model's accuracy) and can be reported separately.
- *Verify.* For every metric, write one sentence naming what it compares against. Check that this is the quantity the method claims to optimize or describe.

**Training on one quantity and evaluating on another.** Training used a soft, probability-weighted proxy, while evaluation used a hard quantity computed against a different target. A reviewer read this as the most serious conceptual failure.
- *Verify.* List the quantity used at each stage (training, model selection, evaluation). If they differ, the paper must say why and show that the difference does not drive the result.

**A formulation with a degenerate optimum.** The objective could be maximized trivially (a region containing a single sample has perfect precision), so the learned solutions collapsed to that trivial case and coverage was near zero. The tables showed it, but it was not noticed before submission.
- *Verify.* Before running full experiments, ask what the cheapest way to maximize the objective is. Check that the termination test, the reward, and the selection step rule it out (minimum support, coverage targets, no reward for exceeding a target, confidence-adjusted scores). Inspect raw outputs early for suspiciously perfect values paired with tiny sizes.

**Wrong or undefined mathematics.** Dimension typos, sign typos, reward terms described only in words, update equations missing a scaling factor, and a term named after a measure it did not compute (a geometric overlap called a divergence). Each one costs credibility far beyond its size.
- *Verify.* Every symbol is defined before use. Every term in an objective has an equation and a numeric weight. Every equation is checked against the code line by line. Quantities are named for what they compute, not for what they resemble.

**Claiming a property the formulation does not have.** The process was described as Markov, but the reward depended on quantities outside the state.
- *Verify.* For a decision process, list every input to the reward and the transition. Each is either in the state or disclosed, with a statement of what the property holds for.

**Theory claims without proof.** The paper described the agents as operating under equilibrium conditions and reported an approximate diagnostic as if it were evidence of convergence. Reviewers rejected both.
- *Verify.* Use the words equilibrium, convergence, optimal, guarantee, and provable only with a proof or a citation that applies to the exact setting. A diagnostic is reported as a diagnostic.

**Attributing a gain to a mechanism without isolating it.** The motivating mechanism (coordination between agents) turned out to have no measurable effect once ablated. The gain came from a simpler factor (having more models per class).
- *Verify.* For every "X improves Y because of Z" statement, there is an ablation that removes Z and keeps everything else. Check confounds such as model count, number of outputs, compute, and gradient updates.

### 1.2 Experimental design

**No held-out evaluation for some results.** Instance-level results used a test split, but class-level results were computed on the full dataset, including training data.
- *Verify.* Fix one split protocol (train, validation, test) for every result. Training and candidate generation use train, every selection and tie-break uses validation, and every reported number uses test. Search the code for any use of test data inside a decision (selection, thresholds, tie-breaking, early stopping).

**Hyperparameters chosen with test data or on part of the seeds.** A weight was set using the test split of one seed, and a threshold was chosen on a subset of seeds.
- *Verify.* Every hyperparameter has a stated selection rule that uses validation data only. If it was chosen on a subset of runs, report the main result on the held-out runs as well.

**Too few and too small datasets for the claim.** Four small datasets (largest about 600 rows) were used to support a claim of scalability.
- *Verify.* Match the claim to the evidence. Use datasets that span sizes, class counts, imbalance, and feature types, including at least one real-world dataset. If the largest dataset is small, do not claim scalability.

**One run per configuration.** The first submission had no seeds, no variability, and no confidence intervals.
- *Verify.* Run every configuration with several seeds (five as a minimum). Report how results vary across seeds and datasets.

**Missing basic reporting.** Model accuracy per dataset, exact hyperparameters, training success rates, runtime, and compute were all absent.
- *Verify.* Use the reproducibility checklist in Part 4.

**Unit and coordinate bugs in reported outputs.** Reported thresholds were physically impossible (negative lengths), because a back-transformation was wrong.
- *Verify.* Add automatic assertions that every reported value lies within the observed range of its feature, in original units. Read a sample of outputs by eye for every dataset before writing.

**Paper and code drift apart.** Several defects were typos in the paper that did not match the code. Others were real differences between what was described and what ran.
- *Verify.* Before submission, audit each equation, constant, and procedure in the paper against the code, and record the file and line that implement it.

**Bugs in the baseline pipeline.** Converting baseline outputs into a comparable form introduced rounding, lost strict inequalities, and allowed empty outputs. All three errors favored the baselines, and fixing them changed the conclusions.
- *Verify.* Audit the baseline pipeline as carefully as your own method. Check edge cases such as empty outputs, ties, rounding, and type conversions.

**Unseeded third-party code.** A library sampled with an unseeded global generator, so its results varied between runs.
- *Verify.* Seed every source of randomness you control. Where you cannot, measure the run-to-run variation, report it, and interpret results whose margin is smaller than that variation with caution.

**Reproducibility across machines.** Runs on different platforms and library versions gave slightly different results.
- *Verify.* Record code version, configuration hash, library versions, and platform with every result file. Re-score stored outputs rather than rerunning when possible.

### 1.3 Baselines

**No fair baseline for the main claim.** The method was proposed for class-level (global) explanation but was compared only with an instance-level method. Reviewers asked for the obvious simple alternative, aggregating the instance-level method across inputs.
- *Verify.* For the main claim, write down the simplest method a skeptical reader would try, and include it. Include a standard strong alternative from a neighboring family (here, a surrogate model). Include a weak floor (random search) to show the metric is not trivially satisfied.

**Unequal budgets.** The proposed method produced up to three outputs per class, while baselines produced one. Part of the apparent gain came from output count.
- *Verify.* Match budgets that affect the metric (number of outputs, candidate pool size, queries, compute). If they cannot be matched, run a matched-budget check and report it.

**Baselines evaluated differently from the method.** Different estimators, different inputs, or self-reported scores make comparisons meaningless (a baseline's own precision estimate was far higher than the shared estimator's).
- *Verify.* Every method is scored by the same code, on the same inputs, with the same estimator. Self-reported numbers are never placed beside shared-estimator numbers.

**Baseline results that depend on a setting you chose.** Aggregated baselines improved with pool size, so the gap depended on that choice.
- *Verify.* Report at least two settings of any baseline parameter that the reader might question, and state the setting used in each comparison.

**Hiding a baseline that wins.** A simple surrogate beat the proposed method on the main metric. Reporting it honestly, and explaining what the proposed method offers instead, was more credible than leaving it out.
- *Verify.* If a baseline beats your method, keep it and state plainly where each method is better.

### 1.4 Metrics

**Metrics not connected to the field's evaluation frameworks.** Reviewers asked how the metrics relate to established properties of explanation quality (correctness, completeness, compactness, contrastivity, stability).
- *Verify.* Map each metric to the property it measures, in one or two sentences, citing the framework. State which properties are not evaluated.

**A composite metric reported without its parts.** A single combined score can hide degenerate behavior (vacuous outputs inflate it, abstention is penalized in a way that favors methods that never abstain).
- *Verify.* Always report the components beside the composite. Say what the composite rewards and penalizes.

**Degenerate values.** Perfect precision on regions containing one sample, or coverage computed on nearly empty outputs, are not evidence.
- *Verify.* Report support (how many samples a value rests on). Flag or filter results with tiny support. Examine outputs that are vacuous (cover almost everything with no information) and wrong (worse than chance).

**Unclear cost accounting.** Claims of cheaper inference need a unit that does not depend on hardware, and a statement of what is counted and what is excluded (training cost, shared precomputation).
- *Verify.* Define the cost unit, count it the same way for every method, report wall-clock time as well, and state what is excluded.

### 1.5 Statistical evaluation

**No statistics in the first submission.** Claims of improvement rested on tables of means.

**Too many tests in the revision.** The first revision added many test families, effect sizes, and intervals, which confused readers.

What worked in the end:
- The unit of analysis is the dataset. Average over seeds within each dataset, then compare methods over datasets.
- Use the paired two-sided Wilcoxon signed-rank test over datasets for two methods (Demšar, 2006). Cite the test and the correction.
- Define one confirmatory family of comparisons for the main metric, fixed before looking at final results. Adjust within it with Holm. Label everything else exploratory.
- Report the mean difference, how many datasets favor each method (for example 9 of 12), and the adjusted p-value. Add one bootstrap interval for the main differences. Skip extra effect-size statistics unless a reader needs them.
- A non-significant difference means the experiment cannot separate the methods. It does not mean they are equivalent.
- Run robustness checks on the main claims. Leave out each dataset in turn (or at least the most extreme one), repeat on runs not used for tuning, and compare against the run-to-run variation of any unseeded component. Report a claim as robust only if it survives these checks.
- Compute every statistic from raw per-run results, never from rounded table values.
- Use the same grouping and correction in the paper and in the analysis code, and keep the analysis code in the repository so reviewers can rerun it.

### 1.6 Claims

**Claims that the tables contradicted.** The abstract and introduction stated improvements that the tables did not show.
- *Verify.* Trace every claim in the abstract, introduction, and conclusion to a table and a test. If a result is mixed, the abstract says so.

**Fragile results presented as firm.** A significant result disappeared when one dataset was removed.
- *Verify.* Soften any claim that fails a robustness check, and say which check it fails.

**Negative results left out.** The revision was more convincing once it stated that the motivating mechanism had no effect and that a simpler baseline was more effective.
- *Verify.* Report negative findings in the abstract or the conclusion when they bear on the main question.

### 1.7 Literature positioning

**Narrow related work.** The section jumped between a few rule learners and the main baseline without situating the work in the field's taxonomy.
- *Verify.* State where the method sits on the standard axes of the field (for explanation methods: local or global, post-hoc or intrinsic, model-agnostic or model-specific, what the objective is). Cite the standard surveys.

**Uncited claims about other methods.** A claim that another family of methods degrades in a certain way had no citation.
- *Verify.* Every statement about another method's behavior has a citation or is removed.

**Novelty claimed on a feature that is not new.** Continuous thresholds were presented as the novelty, although many existing methods already use them.
- *Verify.* For the claimed novelty, list the three closest methods and say in one sentence how yours differs from each. If the difference is not real, find the actual contribution.

**Suggested references ignored.** Reviewers' suggested references must be engaged with in the text, not only cited.

---

## Part 2. How to write the paper

### 2.1 Style rules

- Write short declarative sentences, one idea each. Prefer two sentences to one long sentence joined by a semicolon.
- Avoid colons and semicolons in running prose. They tend to hide lists or loose logic. Use them only in tables, equations, and declarations.
- Write the paper as if it were the first version. Avoid words that reveal a revision history in the manuscript ("now", "no longer", "previously", "in this version", "we changed").
- Avoid filler and words that read as machine-written. Examples are "crucially", "notably", "it is worth noting", "delve", "leverage", "ingredients", "a testament to", "plays a key role", "comprehensive", "robust" (unless tested), "significantly" (unless tested).
- Avoid "This paper asks". Use "We investigate whether" or state the question directly.
- Use one name per concept throughout. Do not alternate between synonyms for the same quantity. When two related quantities exist, name both and use the qualifier every time (for example, one fidelity computed on data rows and one computed on perturbed samples).
- Define a term at first use, and define it once. Do not keep a separate notation table if every symbol is defined in place.
- Give numbers with their context. Say "on average over twelve datasets", give the sample size, and give the support for a value.
- Use the same name for a component in the text, tables, and figures.
- Match the claim strength to the evidence. "Is more effective on all twelve datasets" needs the test. "Comparable" needs a non-significant result and a statement that this is not equivalence.

### 2.2 Structure and content of each section

**Abstract (under 250 words).** Spend the first third on the problem and why existing methods fall short. Then state the idea in one or two sentences, the main results with a few averaged numbers, the most important negative result, and the main limitation. Do not list every result.

**Introduction.**
1. The problem and why it matters, with a concrete example.
2. What existing methods do and where they fall short.
3. The idea and how it addresses the gap.
4. What a good solution must achieve (the requirements the evaluation will test).
5. How the method works, in plain words.
6. The baselines and the main results, with averaged numbers and the main negative result.
7. Three contributions at most. Each is a contribution (a formulation, a method, a finding), not a description of the evaluation.
8. A roadmap that matches the actual section order.

**Related work.** Announce the groups in the opening paragraph and follow them in order, one subsection per group. End with a positioning paragraph that returns to each group in the same order and says how the work differs. Introduce each method once, in the group where it belongs.

**Background.** Only what the method needs, with consistent notation. Give each definition with its equation, and state what the paper later changes.

**Problem formulation.** A separate section that defines the objects and quantities shared by all methods and the evaluation, before any method section.

**Method.**
- A one-paragraph roadmap at the start that matches the subsections.
- Every figure is explained in a paragraph of its own in the text, panel by panel.
- Every equation is followed by a sentence saying what it does and why.
- Every component of an objective has an equation, a weight, and a one-sentence purpose. Related terms (for example, several penalties) go in one numbered equation block.
- State what each design choice prevents (for example, which degenerate solution a constraint rules out).
- Selection rules and thresholds come with the rule that chose them.

**Evaluation metrics.** For each metric, give the question it answers and its formula. Relate the metrics to the field's evaluation framework in one or two sentences. Skip a metric table if the text already defines each one.

**Experimental setup.** Datasets (with sizes, classes, imbalance, model accuracy), splits, models, baselines (with the reason for each), training configuration, and the statistical protocol, including the families of tests.

**Results.**
- Start with the main table and one figure. Put the per-dataset table in the main text, not the appendix.
- Discuss the proposed methods first, then the baselines.
- One table for the main test family, with columns for wins, mean difference, interval, and adjusted p-value.
- A robustness paragraph that reports the checks a skeptical reader would ask for.
- A matched-budget check if budgets differ.
- Example outputs, chosen for being checkable, with the caveats (small samples, label leakage).
- A cost table with queries, time per explanation, and training time.
- No subsection without a finding. Fold small observations into a paragraph.

**Ablations.** One table, one row per removed component, with the main metric and its components. Name components plainly ("several policies per class", not "ingredients"). Say what the ablation can and cannot separate.

**Discussion.** No "Interpretation" subheading. Start directly with what the results mean, then where the method is the better choice and where an alternative is, then the negative results.

**Limitations.** Two paragraphs with only the most important limits, the limits of the evidence (scope, power, baselines, user study) and the limits of the method itself (guarantees, failure modes, uncontrolled comparisons). Do not list every minor caveat.

**Future work.** Concrete next steps that follow from the limitations (more datasets, tuning, other models and training algorithms, other data types). Four items at most.

**Conclusion.** One paragraph that restates the problem, the idea, the main results with their caveats, and the next steps. No new information.

**Declarations.** Use the journal's required list. Check that the data-availability statement is accurate for every dataset.

**Appendix.** Only what a reader needs to reproduce the work or to check a choice, such as implementation details, the selection of key hyperparameters, and per-input results. No notation table, no tables that repeat the text, no software versions that the repository already records. Important results belong in the main text.

### 2.3 Tables and figures

- Captions define the columns and nothing else. Interpretation goes in the text.
- Tables show means over datasets and seeds, and say so in the caption once.
- Put the main comparison in one table rather than spread across several.
- Every figure is referred to and explained in the text.

### 2.4 Writing the response to reviewers

- Keep each reviewer's section separate and self-contained. A reviewer may receive only their own part, so do not refer to answers given to the other reviewer. Repeat the explanation briefly instead.
- Quote each comment verbatim and in full. Use bracketed ellipses only where a sentence is split across two answers.
- Combine comments that raise the same point, and answer them once.
- Start each answer with what changed, not with "We agree" or "Thank you". Thank the reviewer once at the start and once at the end, and where a comment caught an error.
- When something was wrong, state the cause in one sentence (a typo in the manuscript, a code bug, a flawed formulation), then the fix.
- Refer to sections by name, not number. Numbers change with every edit and with the journal class. Avoid equation and table numbers.
- Do not add a separate "changes in the manuscript" line to every answer.
- State plainly where the new results contradict the original claims.

---

## Part 3. A workflow for reaching a good version faster

1. **Write an evaluation plan before running the final experiments.** It lists the claims, the metric for each claim and what it compares against, the baselines and their budgets, the split protocol, the seeds, and the families of statistical tests. Freeze it.
2. **Check for degenerate solutions** of the objective on one small dataset. Inspect the outputs by eye.
3. **Audit the code against the method description**, including the baselines, before the full run.
4. **Run everything with seeds** and store raw per-run results with their configuration, code version, and platform.
5. **Generate a results document by script** from the raw results, with every number the paper will use and every test. Never type numbers by hand.
6. **Write a framing plan** with two lists, claims to make (each with its table and test) and claims not to make (each with the reason).
7. **Write the sections in this order:** problem formulation, method, setup, results, ablations, discussion and limitations, related work, introduction, abstract, conclusion. The introduction and abstract are written last, from the results.
8. **Run an independent audit** of every number and claim against the results document, by a co-author or a separate reviewer who did not write the text.
9. **Review the draft as a hostile reviewer would**, using Part 1 as the checklist.
10. **Do a style pass** using section 2.1, then a compile pass (all references resolved, no placeholders, no undefined citations, no overfull lines).

Do not start writing results text while experiments are still changing. Placeholder numbers and TBD markers multiply the number of drafts.

---

## Part 4. Pre-submission checklist

### Formulation
- [ ] Every symbol is defined before use, and every objective term has an equation and a weight.
- [ ] Every equation matches the code.
- [ ] Each quantity is named for what it computes.
- [ ] The objective has no trivial optimum, or the paper says how it is excluded.
- [ ] Every theoretical property claimed (Markov, convergence, equilibrium, guarantee) is proved, cited, or removed.
- [ ] Training, selection, and evaluation use the same target, or the difference is explained.

### Experiments
- [ ] Train, validation, and test splits are used as stated. No test data enters any decision.
- [ ] Every hyperparameter has a selection rule on validation data.
- [ ] Several seeds per configuration, with variability reported.
- [ ] Datasets match the strength of the claims (size, diversity, at least one real-world dataset).
- [ ] Model accuracy per dataset, hyperparameters, success rates, runtime, and compute are reported.
- [ ] Reported outputs are checked for valid ranges and units.
- [ ] Randomness is seeded, or its run-to-run variation is measured and reported.
- [ ] Code version, configuration, and platform are recorded with each result.

### Baselines
- [ ] The simplest alternative a skeptic would try is included.
- [ ] A strong alternative from a neighboring family is included.
- [ ] A weak floor is included.
- [ ] Budgets are matched, or a matched-budget check is reported.
- [ ] Every method is scored by the same code, inputs, and estimator.
- [ ] The baseline pipeline has been audited for bugs.
- [ ] Sensitivity to the main baseline parameter is reported.

### Metrics
- [ ] Each metric compares against the right target.
- [ ] Each metric is mapped to the field's evaluation framework.
- [ ] Composite metrics are reported with their components.
- [ ] Support is reported, and degenerate values are flagged.
- [ ] Cost units and exclusions are stated.

### Statistics
- [ ] The unit of analysis is stated (usually datasets).
- [ ] One confirmatory family, fixed in advance, with Holm correction. Everything else is labeled exploratory.
- [ ] Tests and corrections are cited.
- [ ] Wins out of N, mean differences, and adjusted p-values are reported.
- [ ] Robustness checks are run on the main claims (leave one dataset out, held-out runs, baseline variability).
- [ ] All statistics are computed from raw results by code in the repository.

### Claims and writing
- [ ] Every claim in the abstract, introduction, and conclusion is traced to a table and a test.
- [ ] Fragile results are softened, and negative results are reported.
- [ ] Novelty is positioned against the closest methods.
- [ ] Every statement about other methods is cited.
- [ ] The paper is positioned in the field's taxonomy with the standard surveys.
- [ ] Captions define columns only. Every figure is explained in the text.
- [ ] No colons or semicolons in prose, no revision language, no filler words.
- [ ] Limitations fit in two paragraphs and cover the most important limits.
- [ ] No placeholders. All references and citations resolve.
- [ ] The data and code availability statements are accurate.
