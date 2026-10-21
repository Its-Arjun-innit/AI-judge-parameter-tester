# Research Methodology: How LLMs Respond to Controlled Changes in Legal Evidence

*Status: draft v0.1 (2026-10-22). This document is frozen into the preregistration after the pilot. The source log is in [sources.csv](sources.csv).*

## 1. Research question and claim boundary

**Research question.** An LLM acts as a Sessions Court judge applying Indian criminal law (BNS, BSA and BNSS, all 2023) to fictional cases. How do controlled changes to one evidence item change the model's verdict and its stated probability that the accused committed the offence? The study varies five things about an item:

- (a) its category;
- (b) its credibility attributes;
- (c) its wording;
- (d) its combination with corroborating or contradicting items;
- (e) its position.

It also asks how stable these responses are across repeated runs and across models.

**Claim boundary.** We measure *observable response sensitivity* to controlled input changes. We do not observe internal reasoning or internal evidence weights. Model-written explanations are data about what the model says, not evidence of why it decided (Turpin et al., 2023). Self-reported probabilities and confidence are treated as responses, not as calibrated probabilities (Kadavath et al., 2022; Xiong et al., 2024). The platform is a research instrument, not a decision tool.

## 2. Literature review

### 2.1 LLMs as legal decision-makers

Three recent studies place LLMs in the decision-maker's role.

- **Posner & Saran (2025).** They replicated a 2×2 experiment originally run on US federal judges (Spamann & Klöhn, 2016) using GPT-4o. The model followed precedent and ignored defendant sympathy. That resembles the law students in the original study, not the judges.
- **Sun et al. (2025).** They replicated a mock-jury experiment with GPT-4o, Claude 3.5 Sonnet and o1, manipulating defendant race, social status, number of allegations and reporting delay. The models gave higher mean guilt probabilities than humans but convicted far less often. They showed some human-like biases, and they responded to reporting delay where humans did not.
- **Bessis et al. (2026).** They tested GPT-4o and GPT-5.2 on simplified criminal cases, manipulating the order of incriminating and exonerating evidence. They found robust order effects (weaker in GPT-5.2) that prompt engineering did not reliably remove.
- **Legal fact prediction (arXiv 2409.07055).** This work separately reports that the party whose evidence is presented last is favoured.

**Implication.** Our order hypothesis (H5) is therefore a **replication and extension** across providers, cases and an Indian procedural setting. It is not a novel finding.

### 2.2 LLM legal capability and judgment prediction

**Capability benchmarks** test legal knowledge and reasoning, not how evidence is weighed:

- LegalBench (Guha et al., 2023);
- bar-exam performance (Katz et al., 2024);
- hallucination profiling (Dahl et al., 2024).

**Judgment prediction** on real cases predicts outcomes from case text, but it confounds evidence evaluation with memorisation of real outcomes:

- ECtHR: Aletras et al. (2016); Chalkidis et al. (2019);
- India: ILDC (Malik et al., 2021); IL-TUR (Joshi et al., 2024).

**Implication.** Fictional cases with researcher-controlled evidence avoid that confound.

### 2.3 Human evidence evaluation

Juror and judge psychology supplies directional predictions and comparison points for our hypotheses:

- **Category (H1).** Confessions are uniquely potent compared with eyewitness and character evidence, even when they are discredited (Kassin & Neumann, 1997).
- **Statistical evidence.** People resist "naked" statistical evidence (Wells, 1992). They also misread match statistics, as in the prosecutor's and defence attorney's fallacies (Thompson & Schumann, 1987).
- **Wording (H2).** Equivalent DNA statistics persuade differently depending on their format (Koehler, 2001).
- **Order (H5).** Belief updating shows order effects that depend on the response mode (Hogarth & Einhorn, 1992).
- **Combination (H3/H4).** Evidence is integrated into coherent stories rather than item by item (Pennington & Hastie, 1992; Simon, 2004).

**Normative benchmark for combination (H3).** Schum's (1994) analysis of source credibility gives the attribute taxonomy: veracity, objectivity, observational sensitivity. Bayesian-network models of legal evidence (Fenton, Neil & Lagnado, 2013) supply the benchmark. When items are conditionally independent, their log-likelihood ratios add, so a zero interaction on the log-odds scale is the normative reference.

### 2.4 Prompt sensitivity and repeatability

LLM outputs shift with factors that should not matter:

- **Formatting:** Sclar et al. (2024).
- **Ordering of examples and options:** Lu et al. (2022); Pezeshkpour & Hruschka (2024).
- **Position within long contexts:** Liu et al. (2024).
- **Position bias in LLM-as-judge settings:** Zheng et al. (2023).

Outputs are also not reproducible even under nominally deterministic settings on hosted APIs (Atil et al., 2024). Together these motivate our design choices:

- the order-randomisation and neutral-item controls;
- prompt-template robustness checks;
- treating run-to-run variability as something to measure (H6).

### 2.5 Counterfactual testing and bias auditing

- **Minimal-pair and contrast-set testing** is a standard way to probe model behaviour: CheckList (Ribeiro et al., 2020); counterfactually augmented data (Kaushik et al., 2020); contrast sets (Gardner et al., 2020).
- **The closest methodological template** is Tamkin et al. (2023), which uses templated decision prompts with a single attribute varied and mixed-effects analysis.
- **Bias audits in legal and quasi-legal decisions** find name-based disparities (Salinas, Haim & Nyarko, 2024). They also find covert dialect prejudice that extends to conviction and death-sentence choices (Hofmann et al., 2024).

**Implication.** Bias is out of scope for the main study and planned as a follow-up. Our cases therefore use role labels only (A5 below).

### 2.6 LLMs as simulated subjects

Work on LLMs as stand-ins for human participants shows both promise and limits: Aher et al. (2023); Binz & Schulz (2023); Dillion et al. (2023). We make no claim that LLM responses predict human judges.

### 2.7 Statistical method

- **Stimuli as a random factor.** When stimuli (here, cases) are sampled, they must be treated as a random factor (Judd, Westfall & Kenny, 2012).
- **Power ceiling.** In such designs, power levels off below 1 however many responses are collected per stimulus (Westfall, Kenny & Judd, 2014). For us, this means **more runs cannot replace more cases**.
- **Supporting methods:**
  - random-effects structure: Barr et al. (2013);
  - equivalence testing for interpretable nulls: Lakens (2017);
  - multiplicity control: Holm (1979); Benjamini & Hochberg (1995).

## 3. Research gap

Existing LLM judge and juror studies manipulate characteristics of the **defendant or case** (race, sympathy, reporting delay, precedent) or the **order** of evidence.

We found no study that does the following:

1. Manipulates the **category and credibility attributes of individual evidence items one at a time**, holding the target fact constant.
2. Tests **evidence combination** (corroboration and contradiction) against a normative log-odds-additive benchmark.
3. Does this across **several providers** with **cases as a random factor** and with pre-specified multiplicity control.
4. Works within **Indian evidence law**, where admissibility rules give a natural probe: a confession to police is inadmissible under BSA s.23, so we can ask whether the model sets it aside.

*This gap claim is provisional until the forward-citation search in sources.csv is complete.*

## 4. Hypotheses

Each hypothesis is something to test, not a conclusion. The primary outcome is the shift Δ in stated probability (percentage points, pp) between paired conditions. The secondary outcome is the verdict (convict / acquit).

| | Hypothesis | Operationalisation | Confirmatory test |
|---|---|---|---|
| H1 | Evidence categories produce different shifts | Category swap: the same disputed fact is established by eyewitness, CCTV, forensic, judicial-confession or documentary evidence. Direction, target fact and stated reliability are held fixed. | Omnibus test of category on case-level Δ; Holm-corrected pairwise contrasts |
| H2 | Wording of equivalent evidence matters | Hedged vs assertive; verbal vs numeric certainty; frequency vs percentage format | Paired Δ ≠ 0; TOST equivalence test at ±5 pp |
| H3 | Corroboration has an effect beyond the individual items | 2×2 design: A present/absent × independent corroborating B present/absent. H3b compares an independent B with a dependent B. | A×B interaction on logit(P) ≠ 0 |
| H4 | The effect of contradiction depends on source and consistency | Contradicting C added; C's credibility (high/low) × consistency with the other evidence | Credibility and consistency effects on Δ |
| H5 | Order matters with the evidence set held fixed | Focal prosecution item placed first vs last; defence evidence always last | Paired Δ(last − first) |
| H6 | Identical runs vary | Same prompt R times at the default temperature, and at T=0 | Within-cell SD, flip probability, Fleiss' κ, ICC (reported as magnitudes) |

**Validity checks** (not hypotheses):

- **Monotonicity:** adding inculpatory evidence should not lower P.
- **Manipulation check:** whether the manipulated item is cited.
- **Implied conviction threshold:** the P at which convict and acquit are equally likely.
- **Inadmissible-confession probe** (exploratory).

## 5. Experimental variables

| Role | Variables |
|---|---|
| Manipulated | Evidence category; attributes (reliability, independence, specificity, source credibility, admissibility, potential bias); wording variant; presence (add/remove); relation (corroborates/contradicts); position |
| Random factors | Case (unit of generalisation); run (residual) |
| Fixed factor | Model (one model per provider: Anthropic, OpenAI, Google) |
| Held constant | System prompt and role; law text; standard of proof; output schema; labelling scheme (PW-n, DW-n, Ex.P-n, MO-n); unrelated facts; the accused's BNSS s.351 statement; generation parameters; demographics (role labels only) |
| Recorded | Position of every item; prompt length; timestamp; model version returned by the API; tokens; errors |
| Controls | A neutral-item condition (separates "more text" from "this evidence"); counterbalanced insertion position; order randomised within each side for H1–H4 |

**Assumptions:**

- **A1.** One stateless API call per trial.
- **A2.** The primary measure is P (0–100), the probability that the accused committed the charged offence. The verdict and confidence (1–7) are recorded separately.
- **A3.** Facts are fictional; the law is real, and the relevant provisions are stated in the prompt.
- **A4.** Evidence is treated as admitted unless admissibility is the manipulated variable.
- **A5.** No names, pronouns, or caste, religion or gender markers.
- **A6.** All materials are in English.

## 6. Sample design

1. **Calibration mini-pilot.**
   - 4–6 candidate cases, baseline condition only, × 3 models × 10 runs.
   - Keep the cases whose baseline P is 30–70 for every model, so that shifts are not hidden by floor or ceiling effects.
2. **Pilot.**
   - 4 cases: 2 resting mainly on direct evidence and 2 circumstantial.
   - About 10 conditions each, × 3 models × 20 runs, for roughly 2,400 calls.
   - Goals: parse-failure and refusal rates; cost and latency; variance components (case-level SD of each effect, run-level SD); manipulation checks.
3. **Power analysis.**
   - Simulate data from the pilot's variance components.
   - Choose the number of cases K and runs per cell R that give 80% power for a smallest effect of interest (SESOI) of 5 pp at Holm-adjusted α.
   - Because of the power ceiling (§2.7), **K is the main lever**. Our provisional expectation is K ≈ 20–40 and R ≈ 10–20.
4. **Freeze and preregister on OSF.** The following are frozen: case content hashes, prompt template versions, model snapshot IDs, the condition list, and the analysis-script commit. A deviations log is kept.

## 7. Statistical analysis plan

**Data and unit of analysis**

- **Data:** long format, one row per trial. Runs are nested in case × condition × model cells.
- **Unit of analysis:** cases are the unit of generalisation. Runs are repeated measurements, never independent cases.

**Primary analysis (two-stage)**

1. Compute cell means.
2. Compute case-level paired differences: Δ = mean P(condition) − mean P(reference).
3. Infer across cases with a sign-flip permutation test and a case-cluster bootstrap 95% CI.

**Confirmatory models**

- **Probability:** linear mixed model `P ~ condition × model + (1 + condition | case)`. If it doesn't converge, simplify the random-effects structure and report that.
- **Verdict:** GEE logistic regression clustered by case, with a cluster bootstrap when K < 30.

**Hypothesis-specific analyses**

- **H3:** the per-case interaction contrast on logit(P), with P clipped to [0.5, 99.5].
- **H2 and any null result:** a TOST equivalence test at ±5 pp.
- **H6:**
  - within-cell SD;
  - modal-verdict agreement;
  - Fleiss' κ, with cells as subjects and runs as raters;
  - ICC from the variance components;
  - at T=0, the share of cells with any variation.
- **Model differences:** the condition × model interaction.

**Effect sizes**

- Δ in pp with a 95% CI (the headline figure);
- d_z for paired comparisons;
- odds ratios for verdicts;
- flip rates.

**Multiplicity**

- **Confirmatory:** one primary test per hypothesis (H1–H5), Holm-corrected.
- **Exploratory** (per attribute, category pair or model): Benjamini–Hochberg at q = .05, labelled as exploratory.

**Exclusions** (fixed in advance)

- An API failure after the maximum number of retries counts as missing.
- Refusals and invalid outputs are outcomes. They are reported per condition and never silently dropped, and they are never re-asked, because re-asking would bias the data towards the conditions that fail.

## 8. Known limitations (to carry into the paper)

- **Small case sample.** Findings generalise only to cases like the ones we sampled. K is small compared with the space of possible cases.
- **No open-weight model.** Exact replication depends on the API models remaining available.
- **Unobservable provider layers.** Provider-side system prompts and safety layers can't be seen.
- **The law is newer than most training data.** BNS and BSA (2023) post-date much of the training data. Models may apply the old IPC and IEA provisions; we code this as a secondary outcome.
- **Category is entangled with specificity.** CCTV is naturally more specific than an eyewitness, so H1 measures "category in its typical presented form".
- **No human comparison.** Comparison with human decision-makers relies on published studies; no new human data is collected.
