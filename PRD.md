# PRD: Hierarchical Segment Label Consistency in Music Structure Analysis

**Target Venues**: TISMIR (primary), ISMIR (conference track)  
**Paper Type**: Reproducibility / Measurement study  
**Status**: Pre-implementation

---

## 1. Problem Statement

Music structure analysis (MSA) systems partition audio into semantically coherent sections and assign labels (Verse, Chorus, Bridge, etc.). The SALAMI dataset — the field's dominant benchmark — provides annotations at two granularity levels:

- **Coarse (MIREX-style)**: broad structural segments widely used in the literature
- **Fine-grained**: subtler subdivisions (e.g., Verse A, Verse B) encoding within-section variation

Almost every published system and evaluation in the last decade trains and evaluates exclusively on the coarse level. This creates a compounded measurement problem: the evaluation score a system achieves is entangled with which annotation level was chosen, yet no published work has systematically quantified this entanglement.

The core questions this paper answers:

1. How consistent are SALAMI's coarse and fine annotation levels with each other?
2. Which tracks and structural categories exhibit the highest inter-level disagreement?
3. How much does evaluation score shift when a boundary detector is trained or evaluated against each annotation level separately?
4. Can variation markers (A/B suffixes) in fine labels predict perceptually important sub-segment distinctions?
5. Does class-balancing the training set (to reduce over-representation of dominant labels such as "Verse") change what the model learns and what the metrics report?

---

## 2. Motivation and Novelty

### Why this hasn't been done

MSA evaluation is notoriously dependent on annotation choices (tolerance window, label vocabulary, annotator). Yet the *vertical* dependency — between the two annotation levels within the same annotator pass — has received no dedicated treatment.

### Specific contributions

| Contribution | Description |
|---|---|
| Inter-level agreement analysis | Cohen's κ, boundary F-measure, and segment-label overlap computed between coarse and fine annotations for all dual-annotated SALAMI tracks |
| Disagreement taxonomy | Categorization of disagreement types: boundary-only, label-only, both |
| Level-conditional benchmark | Boundary detection models trained and evaluated on coarse vs. fine annotations separately; delta in P/R/F reported per model family |
| Variation-marker semantic study | Statistical test of whether A/B suffix distinctions in fine labels correspond to audio feature differences |
| Class-balance ablation | Training set resampling to equalize label frequency; impact on per-class recall and macro F |

### Why TISMIR / ISMIR

- TISMIR accepts measurement and reproducibility papers with rigorous empirical methodology
- The paper makes no new claims about model superiority; its claim is that the field's benchmark is under-characterized
- Directly relevant to MIREX evaluation campaigns

---

## 3. Dataset

### SALAMI (Structural Analysis of Large Amounts of Musical Information)

- **Source**: Public GitHub repository — `DDMAL/salami-data-public`
- **Size**: ~1,400 tracks (audio sourced separately; annotations are the primary artifact)
- **Annotation format**: Plain-text `.txt` files with timestamps and hierarchical labels at two levels per annotator
- **Audio**: Not included in the repo; sourced from the Internet Archive (some tracks) or local copies
- **Dual-annotated subset**: ~200 tracks have both a coarse and a fine annotation from the same annotator; these are the primary analysis corpus

### Audio features (for variation-marker study)

We extract standard MIR features using `librosa`:
- Chroma (12-dim, hop 512)
- MFCC (20 coefficients)
- Tempogram
- Mel-spectrogram (128 bins)

These are used only for the semantic proximity analysis (Contribution 4), not for training boundary detectors.

---

## 4. Methods

### 4.1 Inter-Level Agreement Metrics

For each track with both annotation levels:

- **Boundary agreement**: Compute boundary F-measure (F_b) between coarse and fine boundary sets using a ±0.5s tolerance window (standard in MSA evaluation)
- **Label agreement**: After boundary merging, compute Cohen's κ over the coarser label vocabulary (fine labels are mapped to their coarse parent by stripping suffixes A/B/C)
- **Segment-level IoU**: For each coarse segment, compute maximum IoU against all fine segments; report mean and distribution

Report per-genre breakdown (SALAMI provides genre metadata).

### 4.2 Disagreement Taxonomy

Classify each track into:
- **High agreement** (F_b > 0.8, κ > 0.7)
- **Boundary disagreement** (F_b < 0.5, κ > 0.7): coarse and fine place boundaries differently but agree on labels when boundaries do align
- **Label disagreement** (F_b > 0.8, κ < 0.5): boundaries mostly agree but label assignment diverges
- **Full disagreement** (F_b < 0.5, κ < 0.5)

Visualize with scatter plot and example waveform + annotation overlays for each quadrant.

### 4.3 Level-Conditional Boundary Detection

**Models** (two families, both standard in MSA literature):

1. **Spectral Clustering baseline** (Foote novelty curve): no learned parameters, purely unsupervised; used to test whether the *evaluation* level alone drives score differences
2. **CNN boundary detector**: small ConvNet on Mel-spectrogram patches, trained with binary cross-entropy on boundary/non-boundary frames

**Conditions** (2 × 2 factorial):

| Train level | Eval level | Condition name |
|---|---|---|
| Coarse | Coarse | CC (standard MIREX setting) |
| Fine | Fine | FF |
| Coarse | Fine | CF (train/eval mismatch) |
| Fine | Coarse | FC (train/eval mismatch) |

**Metrics**: Precision, Recall, F-measure at ±0.5s and ±3.0s tolerance; Pairwise F-measure (Pw-F); Normalized Conditional Entropy of clusters (NCE).

Report delta between CC and FF as the primary measurement of how much the annotation level choice inflates or deflates reported scores.

### 4.4 Variation-Marker Semantic Study

Hypothesis: segments labeled "Verse A" and "Verse B" in the fine annotations differ more in audio feature space than two adjacent "Verse" segments in the coarse annotations.

**Method**:
- For each track with A/B distinctions, extract mean feature vector per segment
- Compute cosine distance between all same-label pairs: same-suffix (A–A) vs. different-suffix (A–B)
- Wilcoxon signed-rank test: H0 = no difference in within-type vs. cross-type distances

### 4.5 Class-Balance Ablation

SALAMI label distributions are heavily skewed (Verse and Chorus dominate). Ablation:

1. **Unweighted**: standard training
2. **Inverse-frequency weighted loss**: weight each training sample by 1 / class_frequency
3. **Resampled to cap**: randomly undersample majority classes to max N instances per class

Report per-class recall, macro F, and micro F for the CNN boundary detector under each condition and both annotation levels.

---

## 5. Evaluation Protocol

- **Dataset splits**: 80/10/10 train/val/test, stratified by genre and annotation level
- **Cross-validation**: 5-fold on the dual-annotated subset for agreement metrics (no model training needed here)
- **Seeds**: fixed seeds {42, 123, 777} for all stochastic experiments; report mean ± std
- **Statistical testing**: all pairwise score comparisons use paired Wilcoxon test with Bonferroni correction

---

## 6. Expected Findings and Hypotheses

| Hypothesis | Expected direction | Novelty |
|---|---|---|
| H1: Boundary F_b between coarse and fine < 0.7 on average | Confirmed | Establishes inter-level inconsistency as a real phenomenon |
| H2: FF > CC in boundary F-measure for fine-trained model | Confirmed | Fine labels have more boundaries → higher recall, potentially lower precision |
| H3: CC–FF delta > 5 pp F | TBD | Main measurement claim |
| H4: Variation markers (A/B) are acoustically distinguishable | Partially confirmed | Motivates keeping fine labels |
| H5: Class balancing improves macro F, hurts micro F | Confirmed | Standard but undocumented in this domain |

---

## 7. Deliverables

### Code

```
segment_consistency/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── data.py            # SALAMI loader, annotation parser, feature extractor
│   ├── agreement.py       # Inter-level agreement metrics (F_b, κ, IoU)
│   ├── model.py           # CNN boundary detector
│   ├── train.py           # Training loop with class-balance options
│   ├── evaluate.py        # MSA metrics (P/R/F, Pw-F, NCE)
│   └── utils.py           # Device, seeding, logging helpers
├── experiments/
│   ├── run_agreement.py   # Phase 1: agreement analysis
│   ├── run_boundary.py    # Phase 2: level-conditional training
│   ├── run_variation.py   # Phase 3: variation-marker study
│   └── run_balance.py     # Phase 4: class-balance ablation
├── configs/
│   └── default.yaml
├── scripts/
│   └── plot_results.py
├── tests/
│   ├── test_data.py
│   ├── test_agreement.py
│   ├── test_model.py
│   └── test_pipeline.py
├── data/                  # gitignored
├── results/               # gitignored
└── .gitignore
```

### Paper artifacts

- Table 1: Inter-level agreement statistics (mean F_b, κ, IoU) broken down by genre
- Table 2: 2×2 level-conditional benchmark results (P/R/F, Pw-F, NCE) for Foote + CNN
- Figure 1: Scatter plot of (F_b, κ) per track, colored by genre, quadrant annotations
- Figure 2: Waveform + annotation overlay examples for each disagreement quadrant
- Figure 3: Label frequency distribution at coarse vs. fine levels
- Figure 4: Per-class recall under three class-balance conditions
- Table 3: Variation-marker semantic study results (Wilcoxon p-values, effect sizes)

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Audio not available for all SALAMI tracks | Use annotation-only metrics (F_b, κ) for full corpus; limit audio-dependent experiments to tracks with accessible audio |
| Small dual-annotated subset (~200 tracks) | Report confidence intervals; use cross-validation; frame as feasibility study motivating larger annotation effort |
| Negative results (levels are actually consistent) | Paper's contribution is the measurement itself; null result is publishable at TISMIR as "annotation quality validation" |
| CNN baseline underperforms state-of-the-art | Explicitly position as a controlled measurement probe, not a SOTA claim |

---

## 9. Timeline

| Milestone | Target |
|---|---|
| Dataset acquisition and annotation parser | Week 1 |
| Agreement metrics + Figure 1/2/3 | Week 2 |
| CNN boundary detector + level-conditional evaluation | Weeks 3–4 |
| Variation-marker study | Week 5 |
| Class-balance ablation | Week 6 |
| Paper draft | Weeks 7–8 |
| Internal review and revision | Week 9 |
| TISMIR submission | Week 10 |

---

## 10. Open Questions (to resolve before implementation)

1. **Audio sourcing**: Do we have local access to SALAMI audio files, or do we limit to annotation-only experiments for the boundary F and label agreement analysis?
2. **Fine annotation coverage**: Exactly how many of the ~1,400 SALAMI tracks have *both* a coarse and fine annotation from the same annotator? The answer determines statistical power.
3. **Label vocabulary normalization**: How do we handle annotator-specific label strings (e.g., "Verse 1", "verse", "V") — do we use an existing normalization or create one?
4. **MIREX tolerance window**: Use ±0.5s (strict) or ±3.0s (lenient) as primary? Recommend reporting both.
5. **CNN architecture**: Use a published MSA baseline (e.g., the architecture from Ullrich et al. 2014 or McFee & Ellis 2014) for reproducibility, or a minimal custom ConvNet?
