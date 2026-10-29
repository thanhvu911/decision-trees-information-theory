# Optimizing Decision Trees with Information Theory

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Topic](https://img.shields.io/badge/Topic-Information%20Theory-2a9d8f)

## For readers

This is a research paper and presentation from **CPS 372 Information Theory** at Franklin &
Marshall College (Fall 2024), written by **Thanh Vu and Nam** as a two-person final project.

It explains how decision trees use **entropy** and **information gain** to choose splits, and
how **feature selection** and **pruning** keep trees from overfitting. It is a survey with
worked examples, not an experiment.

| File | What it is |
|---|---|
| [`paper/decision-trees-information-theory-paper.pdf`](paper/decision-trees-information-theory-paper.pdf) | 18-page paper: entropy and information gain, feature selection, pre- and post-pruning, imbalanced data |
| [`paper/information-in-decision-trees-slides.pdf`](paper/information-in-decision-trees-slides.pdf) | 26-slide class presentation (one slide with a third-party image removed) |
| [`analysis/check_examples.py`](analysis/check_examples.py) | Recomputes every worked example in the paper and slides (see the analysis below) |

**Reading guide.** For the information theory, read section 2 of the paper (entropy and
information gain, with worked examples). For the practical machine-learning side, read the
pruning section, which works through cost-complexity pruning, reduced-error pruning, minimum
impurity decrease and minimum leaf size step by step, using a contact-lens tree.

## What the paper covers

1. **Decision tree basics.** Root, decision and leaf nodes, and the trade-offs: interpretable
   and no feature scaling needed, but prone to overfitting, unstable, and biased toward
   features with many levels.
2. **Entropy and information gain.**
   H(S) = −Σ pᵢ log₂ pᵢ and IG(S, A) = H(S) − Σ (|Sᵥ|/|S|) H(Sᵥ), with worked examples.
3. **Feature selection.** Dropping low-information features to get a simpler tree with the
   same predictive power.
4. **Pruning.** Pre-pruning (maximum depth, minimum samples per leaf or split, maximum leaf
   nodes, chosen by k-fold cross-validation) and post-pruning (cost-complexity with α,
   reduced-error, minimum impurity decrease, minimum leaf size).
5. **Imbalanced data.** Oversampling with SMOTE and random undersampling.

## Independent analysis

This review was written after the course. It recomputes every number in the paper and slides
(`python analysis/check_examples.py`) and reads the claims against standard references.

### What holds up

- **The paper's worked example is right.** For 6 Yes / 4 No, H(S) = 0.971. Splitting into
  (3 Yes, 1 No) and (3 Yes, 3 No) gives a weighted entropy of 0.925 and a gain of 0.046. The
  paper rounds these to 0.97, 0.92 and 0.05.
- **The Weather split on the slides is right.** IG = 1.0 − 0.459 = 0.541.
- **The pruning section is the most useful part.** It separates feature selection (which
  removes inputs *before* training) from pruning (which removes branches, while the same
  feature may still be used elsewhere in the tree). Each pruning method comes with steps,
  benefits and drawbacks. The cost-complexity rule, cost = error + α × size, is stated
  correctly.

### What does not hold up

**1. The Humidity example on the slides is miscounted.** The slides say the High group has
"2 No, 1 Yes" and Normal has "2 Yes", which gives IG = 0.541, the same as Weather. The
table on the same slide has four High rows (2 No, 2 Yes) and two Normal rows (1 Yes, 1 No).
Both groups are 50/50, so **IG(Humidity) = 0**: Humidity tells you nothing in this table.

| Split | Information gain (slides) | Recomputed | Gain ratio |
|---|---|---|---|
| Weather (3 values) | 0.541 | 0.541 | 0.371 |
| Humidity (2 values) | 0.541 | **0.000** | 0.000 |
| Wind (2 values) | not shown | **0.459** | **0.500** |

The table also shows something the paper only mentions. Information gain picks Weather,
but **gain ratio** (C4.5's correction for the many-levels bias the paper lists as a
disadvantage) picks Wind. That would be a strong example to add, because it ties the bias to
an information-theoretic fix.

**2. The abstract promises experiments that are not there.** It says the study "offers
experimental results confirming the suitability of decision trees". The paper has worked
examples and illustrative trees, but no dataset, no trained model and no measured accuracy.
The contact-lens pruning example also uses assumed numbers: it assumes the "Hard" leaf accounts for 99% of one branch's accuracy.
In the well-known contact-lens dataset (Cendrowska, 1987), every patient with a reduced tear
rate gets no lenses, so that branch would be a single leaf rather than the subtree drawn.

**3. Two statements contradict each other or overreach.**
- *Minimum impurity decrease.* The description says that a split below the threshold has
  "affected the performance … so it is advisable not to prune". The step-by-step guide
  directly below it says to prune such splits. The step-by-step version is the standard one.
- *Pre-pruning "performs poorly when exposed to new data"* and post-pruning "demands
  high-quality processors". Both are too strong. Pre-pruning with limits chosen by
  cross-validation (maximum depth, minimum samples per leaf) is standard practice and often
  generalizes well. Post-pruning a single tree is
  cheap.

**4. A missed connection to the course.** Information gain *is* the mutual information
between the split and the label, IG(S, A) = I(Y; A). Saying so would link decision trees to
the core of an information theory course. The Gini range of 0 to 0.5 holds for two classes
only; with k classes the maximum is 1 − 1/k.

**5. Two citations need correcting.**
- *Application of Information Theory to the Construction of Efficient Decision Trees* is by
  C. R. P. Hartmann, P. K. Varshney, K. G. Mehrotra and C. L. Gerberich, *IEEE Transactions on
  Information Theory*, 1982. It is not by Meila and Jordan.
- Safavian and Landgrebe's survey appeared in *IEEE Transactions on Systems, Man, and
  Cybernetics* 21(3):660–674, 1991, not in *Pattern Recognition*.
- Most of the other sources are blog posts. The paper would be stronger citing Quinlan (ID3,
  1986; C4.5, 1993) and Breiman et al. (CART, 1984) directly.

### What this project shows

A clear grasp of how entropy drives tree construction, and a practical, well-organized
account of pruning. The errors are in hand-worked numbers and in claims that were not
tested. The obvious next step is a short notebook that grows an unpruned tree on a public
dataset, prunes it with `ccp_alpha` chosen by cross-validation, and reports the accuracy
before and after. That would turn the survey's claims into measured results.

## Reproduce the check

```bash
python analysis/check_examples.py   # standard library only
```
