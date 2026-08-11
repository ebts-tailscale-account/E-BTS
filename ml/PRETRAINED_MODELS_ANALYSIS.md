# Pretrained models for E-BTS force regression — TDNN, ResNet-18, VGG-16

**Status:** analysis only. No code was written, no model was trained, nothing was run on the
robot. Written 2026-08-08 against `HANDOFF.md` §13–§16 and `ml/`.

**Who this is for:** an engineer who knows contact mechanics and control but is deliberately
learning ML. Every piece of jargon is defined the first time it appears, in *italics*.

**Evidence labelling.** Because the previous modelling round went wrong partly by blurring
these together, every non-obvious claim below carries one of:

| tag | meaning |
|---|---|
| **[REPO]** | measured in this project, with the section of `HANDOFF.md` or the file it came from |
| **[LIT]** | published result, cited at the end |
| **[INFER]** | my reasoning from the two above. Not measured. Treat as a hypothesis with a stated confidence |
| **[ENV]** | a fact about this workstation that I checked by inspecting the filesystem |

---

## 0. Executive summary — the answer, up front

1. **VGG-16 is the wrong model for this problem.** 138 M parameters against ~648 independent
   samples, 89% of them in a fully-connected head that has to be thrown away anyway, no batch
   normalisation (which makes small-data fine-tuning harder), and ~10× ResNet-18's compute at
   our native resolution for no capability we need. Its one real credential — VGG-16 is the
   network in the original GelSight force-estimation work [LIT] — does not carry over, for
   reasons in §5.4. **Recommendation: do not build this.**

2. **"Pretrained TDNN" is close to a category error.** A TDNN is a 1-D convolution *over time*.
   The pretrained TDNNs that actually exist and can be downloaded are speaker-recognition
   models (x-vector, ECAPA-TDNN) that consume 80-dimensional log-mel filterbank frames of
   16 kHz audio at 100 frames/s [LIT]. There is no sense in which those weights mean anything
   for a 25 frames/s sequence of tactile images. A TDNN *architecture* trained from scratch on
   the dip ramp is a legitimate and interesting secondary idea — see §3.6 — but it is not
   "using a pretrained model", and it does not address the thing that is actually limiting us,
   which is spatial, not temporal. **Recommendation: not the next thing to build.**

3. **ResNet-18 is the only one of the three worth an experiment**, and only in a specific
   surgically modified form: first convolution adapted from 3 channels to 1 (+2 coordinate
   channels) by *summing* the pretrained RGB weights, the network **truncated after stage 2**
   (which discards 94% of its parameters — see §4.3), global average pooling replaced by a
   3×4 pool, and a ridge or small linear head on top. Expected gain over the existing
   15 k-parameter from-scratch CNN: **small, and quite possibly not measurable** — see point 5.

4. **The single highest-value thing in this document is not one of the three.** The physics
   here is `F = k(x,y) · δ`, and both factors are separately measurable:
   - `δ` (deformation magnitude) is what the marker tracker already gives, and force was
     measured to be **linear in marker displacement, R² = 0.919, r = 0.959** at a fixed
     location **[REPO §15.3]**;
   - `k(x,y)` (local stiffness) is exactly what `franka/map_offset.py` now measures directly
     **[REPO §16.4b]**.

   A model that multiplies a measured stiffness map by a tracked displacement magnitude has
   on the order of **2–3 free parameters** instead of 11 M, cannot overfit 648 samples, and
   makes translation invariance a *feature* rather than a defect. **This should be the
   baseline that any pretrained network is required to beat**, and my honest expectation
   [INFER, medium-high confidence] is that it will be hard to beat.

5. ⚠ **A statistical result that constrains the whole exercise.** Per-seed R² on the 648-sample
   dataset was measured at **0.52–0.77, sd 0.093** **[REPO §15.2g]**. With a 5-seed ensemble the
   standard error of a mean R² is 0.093/√5 = 0.042, so the sd of the *difference* between two
   architectures is 0.042·√2 = 0.059, and detecting a difference at 80% power needs roughly
   2.8 sd ≈ **ΔR² ≈ 0.16, i.e. ≈ 0.09 N RMSE** [INFER, arithmetic shown, high confidence].
   **Any architecture comparison finer than that is below the measurement noise floor at this
   sample size.** Architecture selection is therefore not where the remaining error lives.
   Data design is (§7.1). This is worth internalising before spending a week on backbones.

---

## 1. The problem, verified against the repo

I re-derived these from the handoff and `ml/` rather than taking the brief on trust. Everything
in this section is **[REPO]** with its source.

### 1.1 The input

| property | value | source |
|---|---|---|
| sensor | Prophesee EVK1 Gen3.1 event camera, 640×480 | §3.3 |
| hardware ROI | `0 7 640 450` → 288,000 px retained (93.75%) | §3.4 |
| frame construction | events accumulated over **40,000 µs**, dense count image | §13.1 |
| why 40 ms exactly | illumination is strobed at 25 Hz by the Arduino driver; 40 ms = exactly one cycle, so frames are comparable regardless of phase | §12.5 |
| dtype / range | **uint8 event counts**, max observed **25–27**, median 1 | §13.1, §15.2f |
| occupancy | **15.2–15.3%** of pixels non-zero | §15.2f |
| polarity | **single**. `bias_diff_off = 0` kills OFF in hardware; verified 923,529 ON / 0 OFF. **The polarity bit carries zero information** | §15.2f |
| content | ~211 tracked markers, radius **9.4 px = 0.75 mm**, nearest-neighbour pitch **2.317 mm ≈ 29 px**, scale **12.6 px/mm**, field extent 40.8 × 30.2 mm | §14.3, §16.3 |
| reference frame | every press has its own out-of-contact **tare** phase (1.0 s ≈ 25 frames) | §14.2 |
| displacement noise floor | 0.26 px = **0.020 mm** | §14.3 |

⚠ **Never rescale the counts.** The gray value *is* the event count; percentile-stretching for
display clipped 0.8% of nonzero pixels and those were precisely the marker cores, where the
sub-pixel centroid precision lives (§13.1).

**What the difference image actually looks like** [INFER from the above, high confidence]:
because counts are non-negative and each marker moves by a few pixels, `dwell − tare` is a
**lattice of ~211 small signed dipoles** — negative where a marker's edge was, positive where
it went — on a mostly-zero background, with dipole magnitude and orientation encoding the local
displacement vector. That is a *very* specific image statistic, and it is worth holding in mind
throughout §2.2: it is not a photograph of anything.

### 1.2 The target and the sample count

| property | value | source |
|---|---|---|
| target | scalar normal force `\|Fz\|` in newtons | §14 |
| previous dataset range | **0.243 – 5.945 N**, sd 0.989 N | §15.2e |
| label imbalance | only **~9% of presses exceeded 3 N** | §16.4 Phase 3 |
| ground truth | Wittenstein HEX21 at 1 kHz, per-press tare; noise floor 0.116 N per sample, ~0.004 N after averaging 1 s | §15.2c |
| presses | **648** = 108 locations × 6 depth levels | §15.2e |
| frames | 91,342 (mean 141/press: 24.9 tare, 67.0 dip, 49.0 dwell) | §15.2f |

⚠ **The independent sample count is 648, not 91,342.** Force is flat during a dwell — the
within-dwell sd (0.1168 N) is *identical* to the F/T noise floor (0.116 N), so the ~50 dwell
frames of one press are 50 replicates of a single (image, force) pair (§14.1a). Replicates
average label noise; they do not add degrees of freedom.

⚠ **And for the purpose of estimating generalisation, it is worse than 648.** With splits
grouped by grid row-band there are only **12 rows** and **4 folds** (§15.2g). The effective
number of independent test groups is 4. That is the real reason CV numbers here bounce around.

### 1.3 The two facts that make this problem unusual

**(a) `F = k(x,y) · δ`, and `k` varies 9.4× across the block** (§16.3). *Where* you press
genuinely changes the answer. This is the source of the translation-invariance problem (§2.1).

**(b) Force is linear in marker displacement even though it is nonlinear in depth**
(§15.3). At one location, fitting `|Fz|` on mean marker displacement gave R² = 0.9193,
RMSE 0.0923 N, and adding a `δ^1.5` term changed R² by less than 1e-4. This is linear
elasticity: the bulk displacement field is proportional to load regardless of how load relates
to indenter depth. **The leading-order physics is linear in the image-derived quantity.** That
is a strong argument against needing a deep nonlinear model at all.

### 1.4 What the from-scratch CNN did, and where it failed

Defined in `ml/train_cnn.py`; results in §15.2g. Faithful description (⚠ §16.3 discards these
as *results*; they remain valid as a *description of behaviour*, which is what we need here):

- Input: difference image `(dwell − tare) / 4.0`, downsampled to **120×160**, plus **2
  coordinate channels** (normalised x and y ramps) → 3 channels.
- Body: 4 conv blocks (5×5 then 3×3, stride 2, BatchNorm, ReLU), widths 8→16→32→32.
- Head: `AdaptiveAvgPool2d((3,4))` — **not** global pooling — then dropout 0.3, then one
  `Linear(32·12 → 1)`.
- ~**15 k parameters**. Loss `smooth_l1` on per-fold standardised targets. AdamW, lr 3e-3,
  weight decay 3e-2, cosine schedule, early stopping on an inner grouped split. ~30 s/run
  on the 2080 Ti.

Scoreboard, strict row-band CV, camera-only, no robot inputs (§15.2g):

| model | RMSE | R² |
|---|---|---|
| predict the mean | 0.989 N | 0.000 |
| depth only (robot, the bar) | 0.692 N | 0.510 |
| marker magnitude + shape (ridge) | 0.669 N | 0.542 |
| CNN, 5-seed ensemble | 0.579 N | 0.657 |
| CNN + marker, averaged | **0.509 N** | **0.734** |

**Where it failed** (§15.2g, §16.2):

- **Fold 4 (rows 9–11, the strapped edge) contributes 69% of all squared error** and every
  model fails there, robot-only included. On rows 0–8 the blend reaches ~0.33 N.
- **Predicted-vs-measured slope 0.576** under cross-validation (intercept +0.610, r 0.825),
  where 1.000 is correct. Error walks from **+0.19 N at the 1.5 mm level to −0.32 N at 4.0 mm**.
  The model compresses everything toward the mean.
- A 24-press hard holdout gave slope 1.080 — but only because it barely reached the high-force
  regime, so that number is not reassurance.

---

## 2. Four cross-cutting issues that decide everything

These apply to *all* candidates. It is more efficient to settle them once than to repeat them
three times.

### 2.1 Translation invariance is a defect here, not a feature

**What the words mean.** A *convolution* slides one small weight kernel over the whole image, so
it computes the same function at every position — that makes the *feature maps* (the stacks of
filter responses) shift with the input, which is called *equivariance*. *Invariance* — the
output not changing at all when the input shifts — comes from what you do at the end. Both
ResNet and VGG were designed for classification, where "a cat is a cat wherever it is in the
frame" is exactly right, so both end with an operation that deliberately throws position away.

**What each one does at that stage, precisely:**

| network | final stage | what it produces at 480×640 input |
|---|---|---|
| **ResNet-18** | `layer4` output → `AdaptiveAvgPool2d((1,1))` → `fc` | feature map `[512, 15, 20]` → **each of the 512 channels is averaged over all 300 spatial positions** → a 512-vector with no position information at all → `Linear(512→1000)` |
| **VGG-16** | `features` → `AdaptiveAvgPool2d((7,7))` → flatten → `Linear(25088→4096)` → … | feature map `[512, 15, 20]` → **adaptively average-pooled down to a fixed `[512, 7, 7]`** → flattened to 25,088 → three FC layers |

(Both verified by reading `torchvision/models/{resnet,vgg}.py` on this machine.)

So the two are *not* the same. VGG-16 keeps a coarse 7×7 spatial grid and flattens it, which
means it is **not** fully translation-invariant — it is position-aware at a 7×7 granularity. Its
problem is different: the flatten forces a fixed 7×7 and then spends 102.8 M parameters on the
first FC layer. ResNet-18's global average pooling (GAP) *is* fully invariant and is the harder
problem of the two.

**Why this matters quantitatively here.** Stiffness varies 9.4× across the block (§16.3), the
usable area is ~36 × 30 mm, and the grid pitch was 3 mm. A model that cannot tell rows 0–2 from
rows 9–11 cannot represent `k(x,y)` at all, and `k(x,y)` is most of the signal: a robot-only
model using nothing but `x, y` reached R² = 0.403 on `run1` (§14.1d).

⚠ **A second, subtler quantitative point specific to ResNet.** The *receptive field* — how much
of the input one output unit can see — at ResNet-18's `layer4` output is **435 px**. (Derivation:
start r=1, j=1; `conv1` 7×7/2 → r=7, j=2; maxpool 3×3/2 → r=11, j=4; `layer1` 4×conv3×3 → r=43;
`layer2` → r=99, j=8; `layer3` → r=211, j=16; `layer4` → r=435, j=32.) At 12.6 px/mm that is
**34.5 mm — essentially the entire 40.8 × 30.2 mm marker field**. So even *before* GAP, a single
unit of ResNet-18's deepest feature map already integrates almost the whole sensor. The 15×20
grid is real but heavily overlapping. For VGG-16 the receptive field at `conv5_3` is 196 px
(15.6 mm) — noticeably more local, which is arguably its one genuine architectural advantage
here. [INFER from standard receptive-field arithmetic; the arithmetic is shown so it can be
checked.]

**The options, honestly compared:**

| option | how | cost | assessment |
|---|---|---|---|
| **A. Coarse pooling instead of GAP** | `AdaptiveAvgPool2d((3,4))` on the 512-channel map → 6,144 features | head is `Linear(6144→1)` = 6,145 params, needs ridge | ✅ This is exactly what the from-scratch CNN already does and it demonstrably works. Direct, cheap, no surprises. |
| **B. Coordinate channels (CoordConv)** | append normalised x and y ramps as extra input channels [LIT: Liu et al. 2018] | widen `conv1` from 3 to 3 input channels; keep pretrained weights for the image channel, initialise the two coordinate columns to **zero** so the network is bit-identical to pretrained at initialisation | ✅ Clean, and the from-scratch CNN already uses it. Zero-init of the new columns is the detail that makes it safe with pretrained weights. |
| **C. Truncate earlier** | take `layer2` output (`[128, 60, 80]`, receptive field 99 px = 7.9 mm) instead of `layer4` | drops 10.5 M of 11.2 M parameters | ✅ Solves the resolution and the over-parameterisation problem in one move. My preferred option (§4.6). |
| **D. Spatial softmax / keypoint layer** | per channel, compute the softmax-weighted expected (x,y) → 2×512 numbers [LIT: Levine et al. 2016, used widely in visuomotor policies] | tiny head | 🟡 Elegant and explicitly positional, but 512 keypoints from ImageNet channels is a strange object; unproven here. |
| **E. Keep GAP, and factorise the physics instead** | let the CNN predict only the translation-invariant quantity (deformation magnitude δ), and get `k(x,y)` from the **measured** stiffness map, then multiply | ~2 extra parameters | ⭐ **The best idea in this section.** Translation invariance stops being a defect because you have deliberately given the invariant part of the problem to the invariant model. See §7.1. |
| F. Rely on padding leaking position | zero-padding does encode some absolute position in CNNs [LIT: Islam et al. 2020] | free | ❌ Real effect, but relying on it is fragile and untestable. Don't. |

⚠ **A deployment caveat on any location-aware option.** At inference the model must get the
location from the *sensor*, not the robot. That is fine — the contact centroid is directly
estimable from the difference image (the tracker already computes `cx_mm` in
`extract_features.py`) — but it must be built that way from the start, or you will have trained
a model that cannot be deployed.

### 2.2 The domain gap: does ImageNet pretraining transfer to a sparse count-image dot lattice?

**What ImageNet pretraining is.** The weights ship having been trained to classify 1.28 M
photographs into 1,000 categories. The transfer hypothesis is that the features learned for that
are reusable elsewhere.

**What the literature actually establishes:**

- **The lowest layers are the general ones.** Networks trained on natural images learn
  Gabor-like oriented edge filters and colour blobs in conv1, and these are not specific to the
  dataset [LIT: Yosinski et al. 2014].
- **On non-natural imagery, the benefit is mostly confined to those low layers.** Raghu et al.
  (2019) evaluated ImageNet transfer on two large medical-imaging tasks and found transfer
  "offers little benefit to performance", that lightweight models matched full ImageNet
  architectures, and — via layer-wise weight-transfusion experiments — that **feature reuse is
  primarily happening in the lowest layers**; much of the apparent benefit was over-
  parameterisation rather than sophisticated reuse [LIT].
- **ImageNet accuracy does predict transfer accuracy** across natural-image tasks, for both
  fine-tuning (r = 0.96) and frozen-feature logistic regression (r = 0.99) [LIT: Kornblith et al.
  2019]. ⚠ But that study is 12 *natural image* datasets. It does not license extrapolation to a
  synthetic dot lattice.
- **Pretraining mainly buys convergence speed, not final accuracy.** He, Girshick & Dollár (2019)
  showed models trained from random initialisation match ImageNet-pretrained ones on COCO given
  enough iterations, and that this held with 10% of the data; pretraining "speeds up convergence
  early in training, but does not necessarily provide regularization or improve final target
  task accuracy" [LIT]. ⚠ Their smallest regimes are still thousands of images, so this is
  *suggestive*, not conclusive, at n = 648.
- **Specifically for event cameras**: ImageNet-pretrained backbones *are* the standard practice
  and *do* help on event classification — pretrained models on N-Caltech101 reach ~85% vs ~72%
  for the same architecture without pretraining [LIT], and Gehrig et al. (2019) build their
  event-representation work on an ImageNet-pretrained ResNet-34 [LIT]. **But** Klenk et al.
  (WACV 2024) pretrained directly on event data (Masked Event Modeling) and concluded that on
  real-world event data their event-domain pretraining is **superior to supervised RGB-based
  pretraining**, and that "the common practice … of simply transferring RGB-pretrained weights
  to the event domain is not always the best option" [LIT].

⚠ **Crucially, none of that event-camera evidence is about our kind of event image.**
N-Caltech101 and N-Cars are *moving natural scenes* — the events trace out object contours, and
an accumulated event frame genuinely resembles an edge map of a photograph. That is why ImageNet
edge filters help there. **Our frames are a static strobed marker lattice**: ~211 discs of fixed
1.5 mm diameter on a 2.3 mm grid, with events generated by the 25 Hz illumination rather than by
scene motion. The resemblance to a natural-image edge map is much weaker.

**My assessment of what would actually transfer** [INFER, with confidence stated]:

| layer | what it encodes on ImageNet | transfers to our input? | confidence |
|---|---|---|---|
| conv1 (7×7 for ResNet, 3×3 for VGG) | oriented edges, blobs, colour opponency | **Partly.** The oriented-edge subset should respond usefully to marker boundaries — the marker is ~19 px across and ResNet's conv1 is 7×7 at stride 2, a sensible scale match. **The colour-opponent subset is dead weight** (§2.2b). | medium-high |
| early-mid (ResNet layer1–2, VGG blocks 1–3) | corners, junctions, simple textures | **Weakly.** Some corner/blob detectors are reusable; most texture units have nothing to respond to on a 15%-occupancy count image. | medium |
| deep (ResNet layer3–4, VGG blocks 4–5) | object parts, natural-image texture statistics | **Essentially not.** ImageNet CNNs are strongly texture-biased [LIT: Geirhos et al. 2019]; there is no natural texture here. And this is where 94% of ResNet-18's parameters live (§4.3). | high |

**(2.2b) The channel-count mismatch, concretely.** The models want `[3, H, W]` float tensors,
normalised with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. We have
one channel. Four options:

| option | what it does | verdict |
|---|---|---|
| **Replicate to 3 channels** | feed `x` as R=G=B | Works, costs 3× the conv1 compute for nothing. **Every colour-opponent filter — one whose weights sum to ≈0 across the RGB axis — outputs ≈0 by construction.** A meaningful fraction of conv1 capacity is silently wasted. |
| **Sum the pretrained conv1 weights over the channel axis** → `[64,1,7,7]` | mathematically **identical** to replicating, but 3× cheaper | ⭐ The standard grayscale adaptation. Do this. The colour-opponent filters become near-zero kernels, which you can then prune or re-initialise. |
| **Average instead of sum** | same shape, but responses scaled by 1/3 | Acceptable if you also rescale the input, but the sum preserves activation magnitude, which matters because the *next* layer is a BatchNorm calibrated on ImageNet statistics. Prefer sum. |
| **Re-initialise conv1 randomly** | 9,408 weights, trivially learnable from 648 samples | Reasonable fallback, and worth running as an ablation. But it discards the one layer that most likely *does* transfer. |

⚠ **A normalisation trap.** ImageNet preprocessing assumes inputs in roughly [0,1] scaled to
zero-mean/unit-variance. Our difference image is signed, extremely sparse (15% occupancy) and
integer-valued in roughly ±25. If you feed it through ImageNet normalisation unchanged, the
first BatchNorm sees statistics nothing like what it was calibrated on. The from-scratch CNN
divides by 4.0 to land near unit variance — do the equivalent, and **check the activation
statistics after conv1 empirically** rather than assuming.

### 2.3 Parameter count versus sample count

The raw numbers:

| model | parameters | params per independent sample (n=648) |
|---|---|---|
| existing from-scratch CNN | ~15,000 | 23 |
| ResNet-18 | 11,689,512 | 18,040 |
| VGG-16 | 138,357,544 | 213,515 |
| ResNet-18 truncated after `layer2` | ~681,000 | 1,051 |

⚠ **"Parameters ≫ samples" is not automatically fatal.** Modern deep networks routinely
generalise in that regime, and the classical p ≫ n intuition from linear regression does not
transfer cleanly. What *is* fatal is **trainable degrees of freedom ≫ samples with no other
source of constraint**. Pretraining is precisely such a source of constraint — the weights start
somewhere sensible and you limit how far they move. So the right question is not "how many
parameters" but "**how many are you actually letting move, and how far**".

The four regimes, in increasing order of risk:

**(a) Frozen backbone + linear probe.** *Linear probe* = run every image through the frozen
network once, keep the resulting feature vector, and fit a linear model (here: ridge regression)
from features → force. Nothing in the backbone changes.

- ResNet-18 with GAP: 512 features vs 648 samples → p/n = 0.79. Ridge-able, on the edge.
- ResNet-18 with 3×4 pooling: 512 × 12 = **6,144 features** vs 648 → p/n = 9.5. Needs strong
  ridge, or PCA down to ~50 components first.
- VGG-16 `fc7`: 4,096 features → p/n = 6.3. Same problem.
- ✅ **This is viable, it is cheap (one forward pass over 648 images, seconds), and it directly
  answers the question "do ImageNet features contain anything about our force?"** It should be
  the *first* thing run, because it is the cheapest possible falsification test.
- ⚠ Ridge is an explicit shrinkage estimator, so expect it to make the slope-compression problem
  *worse*, not better (§2.4).

**(b) Partial fine-tuning of late blocks.** The conventional recipe (freeze early, train late)
is **backwards for this problem**. Late blocks are where the domain gap is largest and where 94%
of the parameters are. Unfreezing `layer4` alone means training 8.4 M parameters on 648 samples,
which is the worst of both worlds. ❌ Do not do this.

**(b′) Partial fine-tuning of *early* blocks, with late blocks discarded.** ✅ This is the
inverted version that actually matches the evidence: keep conv1–layer2 (0.68 M params), initialise
from ImageNet, fine-tune with a low learning rate, and throw `layer3`/`layer4` away entirely.
This follows directly from Raghu et al.'s finding that reuse is concentrated in the lowest layers
[LIT] and from the receptive-field arithmetic in §2.1.

**(c) Full fine-tuning.** ResNet-18: 11.7 M trainable on 648 samples with 4 CV folds. Possible
with heavy augmentation, discriminative learning rates and early stopping, but the *measurement*
problem bites — per-seed R² sd is 0.093 (§0.5), so you cannot tell whether your careful recipe
helped. VGG-16 full fine-tune: 138 M on 648 samples, no BatchNorm to stabilise it. ❌ Not viable
for VGG; 🟡 marginal for ResNet-18 and not where I would spend the time.

**(d) The two-step LP-FT recipe.** Kumar et al. (2022) show that fine-tuning can *distort* good
pretrained features and underperform a linear probe out-of-distribution, and that linear-probing
first and then fine-tuning (LP-FT) beats both — reported as ~1% better in-distribution and ~10%
better out-of-distribution [LIT]. ⚠ **This is unusually relevant here**, because our grouped CV
*is* an out-of-distribution test by construction: fold 4 holds a stiffness regime the model never
saw. If any fine-tuning is done, do LP-FT, not naïve fine-tuning.

**Augmentation — one warning.** Augmentation manufactures images, not labels (§14.1h). And most
standard augmentations are *actively wrong* here: horizontal/vertical flips and translations
change `(x,y)`, and therefore change `k(x,y)`, and therefore change the true force — while
leaving the label untouched. **Flips and shifts would teach the network the exact invariance we
are trying to prevent.** Safe augmentations are limited to: additive count noise consistent with
the Poisson-like event statistics, small intensity scaling, and dropping random events. That is
a much smaller toolbox than usual, and it further reduces the appeal of high-capacity models.

### 2.4 The compression failure mode (slope 0.576) — what it is and what fixes it

**First, a piece of theory that sharpens the diagnosis.** Suppose a predictor outputs the true
conditional mean, `ŷ = E[y|x]`. Then
`Cov(y, ŷ) = E[y·ŷ] − E[y]E[ŷ] = E[ŷ²] − E[ŷ]² = Var(ŷ)`,
so the OLS slope of measured on predicted is `Cov(y,ŷ)/Var(ŷ) = ` **exactly 1.000**, no matter how
weak the model is. A model that explains only 10% of the variance still has slope 1 if it is
*calibrated* — it simply produces a narrow, near-constant prediction.

⚠ **Therefore slope 0.576 is not "the model is weak". It is "the model is over-shrunk".** That is
a diagnosable, separable defect, and it is good news: it means there is recoverable signal being
thrown away. Four mechanisms could produce it here:

1. **Extrapolation under grouped CV.** Fold 4 (rows 9–11) has mean force 2.08 N vs 1.50 N
   elsewhere and max 5.94 N, and the model has never seen that regime when tested on it
   (§15.2g). A model asked to predict outside its training range will regress toward the range
   it knows. Given fold 4 contributes 69% of squared error, **I believe this is the dominant
   term** [INFER, medium-high confidence].
2. **Label imbalance.** Only ~9% of presses exceeded 3 N (§16.4). MSE-style training on an
   imbalanced continuous target is known to bias predictions toward the dense region of the label
   distribution [LIT: Yang et al. 2021, "Delving into Deep Imbalanced Regression"; Ren et al.
   2022, "Balanced MSE"].
3. ⚠ **The loss function, which I think is an underappreciated contributor here.**
   `train_cnn.py:176` uses `F.smooth_l1_loss` on targets standardised to unit sd. Smooth-L1 with
   the default β = 1.0 is quadratic for residuals below 1 and **linear above** — so any error
   larger than one standard deviation (≈ 0.99 N) receives a *constant* gradient. Combined with
   only 9% of samples above 3 N, the tails are down-weighted twice over. The network is being
   explicitly instructed not to chase the extremes. [INFER from reading the code, high confidence
   that this contributes; unquantified how much.]
4. **Weight decay (3e-2) and early stopping**, both of which shrink outputs toward the mean.

**Now, per candidate** (this is the "help / hurt / neutral" the brief asks for):

| candidate | effect on the compression | why |
|---|---|---|
| **Frozen backbone + ridge probe** | ⚠ **hurts** | Ridge is shrinkage by definition; the regularisation that makes 6,144 features tractable on 648 samples is the same thing that pulls predictions toward the mean. Expect slope *below* 0.576. |
| **ResNet-18, fine-tuned** | **neutral** | Same loss, same labels, same folds. Nothing about the architecture touches any of the four mechanisms. Its extra capacity might fit the stiff edge slightly better *if* that regime were represented in training — it isn't, under grouped CV. |
| **VGG-16, fine-tuned** | ⚠ **hurts, mildly** | Requires more regularisation for the same data, and regularisation is mechanism 4. |
| **TDNN over the dip ramp** | 🟡 **could genuinely help — the only one that could** | The dip is the one place where force varies while position is held constant (§14.2). Training on ramp frames turns each press into a *trajectory* spanning 0 → peak force, which **directly attacks mechanism 2**: the high-force region stops being a rare 9% class and becomes the end of every trajectory. It does nothing about mechanism 1. |

**What actually fixes it** (architecture-independent, in order of expected effect):

1. ⭐ **Fix the data.** §16.4 Phase 3 already says design by *force*, not depth — use the measured
   stiffness map to precompute per-location depths that hit target force levels, giving a
   balanced force histogram. That directly removes mechanism 2 and reduces mechanism 1. This is
   the fix; everything below is mitigation.
2. **Fix the strap problem** (§16.4 Phase 0). Removing the 9.4× stiffness range removes the
   extrapolation regime that is mechanism 1.
3. **Change the loss**: plain MSE, or Balanced MSE [LIT: Ren et al. 2022], or LDS/FDS [LIT: Yang
   et al. 2021], or simply raise smooth-L1's β so the linear regime starts further out.
4. **Report slope as a first-class metric** — already decided in §16.4 Phase 2. Add
   intercept and per-decile residual mean alongside it.
5. **Post-hoc recalibration**: fit `y ≈ a·ŷ + b` and invert. ⚠ This must be fitted on a *nested*
   held-out split or it leaks, and it is cosmetics — it improves the slope while leaving RMSE
   roughly unchanged or slightly worse. Use it only after 1–3, and say so in the write-up.

---

## 3. Candidate 1 — TDNN (Time-Delay Neural Network)

### 3.1 What it is

A TDNN is the original convolutional architecture, invented for phoneme recognition (Waibel et
al., 1989) [LIT]. The idea: take a sequence of feature vectors, one per time step; at each layer,
each unit looks at a small **window of consecutive time steps** across all features, and the same
weights are used at every position in time. That is a **1-D convolution along the time axis**,
with the feature dimension playing the role that colour channels play in a 2-D CNN. Because the
weights are shared across time, a pattern learned at one moment is recognised at any other —
*shift invariance in time*.

The original network for /b/, /d/, /g/ took 15 frames × 16 mel-scale spectral coefficients
(10 ms per frame), had a hidden layer of 8 units over a 3-frame context, a second of 3 units over
a 5-frame context, and averaged over time at the output. It was on the order of a few thousand
weights [LIT].

Modern descendants replace the stacked short windows with **dilation** — the window skips time
steps, so a small number of layers covers a long context cheaply (Peddinti et al. 2015) [LIT].
The dominant modern instances are:

- **x-vector** (Snyder et al. 2018): 5 TDNN layers over frames → a *statistics pooling* layer
  that takes the mean and standard deviation over the whole utterance → two dense layers → a
  speaker embedding.
- **ECAPA-TDNN** (Desplanques et al. 2020): 1-D convs, squeeze-and-excitation Res2Blocks,
  **attentive** statistics pooling, 192-dimensional embedding, trained on VoxCeleb1+2 [LIT].

### 3.2 ⚠ What "pretrained TDNN" actually means, and why that is a problem

The downloadable pretrained TDNNs — the SpeechBrain / Kaldi ECAPA-TDNN and x-vector checkpoints —
are **speaker recognition models**. Their input contract is:

| | |
|---|---|
| input | 80-dimensional log mel-filterbank energies [LIT] |
| frame rate | 100 frames/s (25 ms window, 10 ms hop) from 16 kHz audio |
| training data | VoxCeleb1 + VoxCeleb2 speech, plus MUSAN/RIR augmentation [LIT] |
| output | a 192-d embedding trained with an angular-margin speaker classification loss |

Our sequence is 25 frames/s of tactile images with no acoustic content whatsoever. **There is no
mechanism by which weights tuned to distinguish human voices in mel-frequency space could carry
information about silicone deformation.** Unlike ImageNet→event-frames, where at least "oriented
edges are oriented edges" is a coherent argument, here the input space has no shared structure at
all — not the dimensionality, not the sampling rate, not the physical meaning of an axis.

⚠ **So if the operator's plan is "download a pretrained TDNN and fine-tune it", that plan does
not have a target.** What *is* available and sensible is the TDNN **architecture**, trained from
scratch — which is a fine thing to do, but it is the opposite of the decision recorded in §16.4a.
This should be surfaced explicitly rather than discovered during implementation.

### 3.3 What temporal signal exists in this rig

Per press (measured, §15.2f, 91,342 frames / 648 indents):

| phase | frames (mean) | duration | what it contains |
|---|---|---|---|
| `tare` | 24.9 | 1.0 s | out of contact — the reference. Force ≈ 0. |
| `dip` | 67.0 | ~2.7 s | descent **plus** the `dip_to_depth()` closed-loop depth-correction iterations (§4.11). Force ramps 0 → peak at a **fixed location**. |
| `dwell` | 49.0 | 2.0 s | constant force. 50 replicates of one label. |
| **total** | **141** | ~5.7 s | |

The dip is the interesting part and §14.2 already flags it: **it is the only place in the dataset
where force varies while position is held constant** — roughly 43,000 frames across the campaign
that break the location/force confound.

### 3.4 Hard constraints on any temporal model here

1. ⚠ **Effective temporal resolution is 40 ms, full stop.** The illumination is strobed at 25 Hz
   by the Arduino driver (§12.5), and a 40 ms accumulation window spans exactly one cycle. You
   can *slide* the window at 60 fps (as `event_video.py` does) but consecutive frames then
   overlap 58.3% and **are not independent samples** (§13.1). Non-overlapping sampling gives
   **25 Hz, i.e. a Nyquist limit of 12.5 Hz**. Anything faster than that is invisible to this
   pipeline as currently configured.
2. ⚠ **The event *rate* carries essentially no contact information**: `corr(n_events, |peak_Fz|)
   = −0.184` (§14.1f), because the 25 Hz strobe dominates the temporal signal — the 1 ms
   event-rate autocorrelation is r = 0.98 at a 40 ms lag whether pressed or hovering (§12.5).
   **A TDNN fed the scalar event rate over time would learn nothing.** Its input must be a
   per-frame *spatial* summary.
3. **Viscoelasticity is small but real.** Fitting on the loading ramp and testing on unloading
   gave RMSE 0.088 N with bias only −0.025 N (§15.3). So there is a rate/history effect, but it
   is at the level of the noise — limiting the upside of explicitly modelling time.
4. **Motion blur on the ramp.** At ~8 mm/s tool speed, per-frame marker motion is ~1.3 px
   (§14.2) — within what the tracker handles (max step 3.96 px) but not nothing.
5. **The ramp is not quasi-static**, so ramp force at a given depth ≠ the dwell value at the same
   depth (§14.2). Labels along the trajectory are honest instantaneous force readings, which is
   fine, but they are not interchangeable with dwell labels.

### 3.5 What a TDNN would look like here, if built

**Input.** Not the image. A per-frame feature vector `z_t ∈ R^d`, one per 40 ms frame, e.g.
`d ≈ 10–40`: mean/max/RMS marker displacement, contact centroid `(cx, cy)`, radial vs tangential
split, anisotropy, concentration — i.e. the columns already produced by `extract_features.py`.
Then a sequence `Z ∈ R^{T×d}` with `T ≈ 90` (dip + dwell).

**Body.** 3–4 dilated 1-D conv layers (dilations 1, 2, 4), widths 32–64, receptive field ~15–30
frames = 0.6–1.2 s.

**Output — and this is the interesting design choice.** Two options:
- **(i) Per-frame force**, `T` outputs. Turns each press into ~90 labelled examples spanning
  0 → peak force. This is the version that attacks the label-imbalance problem (§2.4).
- **(ii) One scalar per press** via statistics pooling, exactly as x-vector does. Simpler, but
  throws away the reason for using a temporal model in the first place.

⚠ Option (i) needs care: ~90 outputs per press are *not* 90 independent samples, and the
evaluation must still be grouped by location. Statistically it adds label diversity, not sample
count.

**Size.** ~20–100 k parameters. Same order as the existing CNN. No pretrained weights.

### 3.6 Verdict on TDNN

| | |
|---|---|
| As a **pretrained** model | ❌ **Does not exist for this modality.** The only pretrained TDNNs are speech models with an unrelated input contract. |
| As an **architecture** | 🟡 **Legitimate, and interesting for one specific reason** — it is the only candidate that directly attacks the label-imbalance half of the slope-0.576 problem, by turning 648 scalar labels into 648 force *trajectories*. |
| But | ⚠ It does not address the core difficulty, which is spatial (`k(x,y)` varying 9.4×), and it inherits whatever per-frame spatial features you feed it. **A TDNN on bad features is a bad model with extra steps.** |
| Sequencing | Build the spatial model first. Then, if you want a force *trajectory* output or need to handle rate-dependence, wrap it in a temporal model. Not before. |

---

## 4. Candidate 2 — ResNet-18

### 4.1 What it is, mechanically

*ResNet* = "residual network" (He et al., 2016). Its innovation is the **skip connection**: each
block computes `y = x + F(x)` rather than `y = F(x)`, so the block only has to learn a *correction*
to its input. This makes deep networks trainable — without it, gradients vanish through many
layers.

ResNet-18, exactly as torchvision defines it (read from
`torchvision/models/resnet.py` on this machine):

| stage | operation | output at 480×640 input | parameters |
|---|---|---|---|
| `conv1` | 7×7 conv, 64 filters, stride 2, pad 3 | `[64, 240, 320]` | 9,408 |
| `maxpool` | 3×3, stride 2 | `[64, 120, 160]` | 0 |
| `layer1` | 2 BasicBlocks, 64 ch, stride 1 | `[64, 120, 160]` | 147,456 |
| `layer2` | 2 BasicBlocks, 128 ch, first stride 2 | `[128, 60, 80]` | 524,288 |
| `layer3` | 2 BasicBlocks, 256 ch, first stride 2 | `[256, 30, 40]` | 2,097,152 |
| `layer4` | 2 BasicBlocks, 512 ch, first stride 2 | `[512, 15, 20]` | 8,388,608 |
| `avgpool` | `AdaptiveAvgPool2d((1,1))` — **this is the GAP** | `[512]` | 0 |
| `fc` | `Linear(512 → 1000)` | `[1000]` | 513,000 |
| | (+ BatchNorm parameters) | | ~9,600 |
| **total** | | | **11,689,512** ✓ |

A *BasicBlock* is: 3×3 conv → BatchNorm → ReLU → 3×3 conv → BatchNorm → add the input → ReLU.
*BatchNorm* normalises each channel using batch statistics; it is what makes these networks
tolerate large learning rates, and it is a genuine advantage over VGG for small-data fine-tuning.

Reference numbers: 69.76% ImageNet top-1, ~1.8 GFLOPs at 224×224 (torchvision).

### 4.2 What its input must be

- Shape `[B, 3, H, W]`, float32. `H, W` ≥ 33 or so (GAP makes it size-agnostic); ImageNet
  evaluation uses 224×224 but nothing forces that.
- Normalised with mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- For us: sum the conv1 weights across the channel axis → `[64, 1, 7, 7]` (§2.2b), optionally
  widen back to 3 to append two zero-initialised coordinate channels (§2.1 option B).

### 4.3 ⚠ Where the parameters actually are

| stage | parameters | share of the 11.17 M convolutional total |
|---|---|---|
| conv1 | 9,408 | 0.08% |
| layer1 | 147,456 | 1.3% |
| layer2 | 524,288 | 4.7% |
| **layer3** | **2,097,152** | **18.8%** |
| **layer4** | **8,388,608** | **75.1%** |

**94% of ResNet-18's convolutional parameters live in `layer3` and `layer4` — exactly the two
stages least likely to transfer to a non-natural domain (§2.2), and the two whose receptive
fields (211 px and 435 px = 16.7 mm and 34.5 mm) already exceed or nearly equal the whole marker
field.** This is the single most decision-relevant fact about ResNet-18 for this problem
[INFER from the arithmetic above; the arithmetic reconciles exactly with the published
11,689,512 total, so I am confident in it].

**Consequence:** truncating after `layer2` gives a **0.68 M-parameter** feature extractor with a
99 px (7.9 mm) receptive field — a much better match to a deformation field with a measured
~8 mm half-width (§16.3) — while discarding 94% of the capacity you cannot afford anyway.

### 4.4 How it would be applied here, concretely

Recommended configuration [INFER, this is my design proposal, not something measured]:

```
input   : difference image (dwell − tare), 1 channel + 2 coordinate channels
resolution: 240×320 (half of native) — see §4.5 for why not full res
conv1   : pretrained 7×7 weights summed over RGB → [64,1,7,7]; two extra input
          columns for the coordinate channels, initialised to ZERO
layer1-2: pretrained, either frozen or fine-tuned at 0.1× the head learning rate
layer3-4: DISCARDED
head    : AdaptiveAvgPool2d((3,4)) on the [128, 30, 40] map → 128×12 = 1,536 features
          → dropout → Linear(1536 → 1)
```

Trainable parameters: ~1,537 (frozen) to ~683,000 (early layers unfrozen). Both are defensible
against 648 samples; the second only with LP-FT (§2.3d) and heavy early stopping.

**Sequence of experiments, cheapest first** (each falsifies something):

1. **Frozen linear probe, no coordinate channels, GAP.** 512 features → ridge. Runtime: one
   forward pass over 648 images, seconds. **If this scores worse than predict-the-mean, ImageNet
   features contain nothing about our force and you can stop.**
2. Frozen probe with 3×4 pooling (6,144 features → PCA to 50 → ridge). Tests whether keeping
   position helps, in isolation.
3. Add coordinate channels. Tests whether explicit position helps beyond pooled position.
4. Truncate at layer2, unfreeze, LP-FT.
5. Only if 1–4 show a trend: full fine-tune, 5 seeds.

⚠ Everything under strict row-band grouped CV, 5 seeds, reporting RMSE **and slope**, against the
predict-the-mean and depth-only baselines from §15.2g.

### 4.5 Resolution, memory and time on the 2080 Ti

| input | ResNet-18 forward FLOPs/image | activation memory/image (fp32, approx) | comment |
|---|---|---|---|
| 224×224 | 1.8 G | ~25 MB | ImageNet default. **Throws away most of our spatial detail**: 640→224 is a 2.9× reduction, taking the 29 px marker pitch down to ~10 px and the 0.26 px displacement noise floor to ~0.09 px equivalent. |
| 240×320 | 2.8 G | ~35 MB | Marker pitch ~14 px. Reasonable compromise. |
| 480×640 | 11 G | ~140 MB | Native. Batch 32 ≈ 4.5 GB — fits in 11 GB. |

[INFER: FLOPs scaled by pixel count from the published 1.8 GFLOPs @224; activation memory
estimated by summing feature-map sizes. Both approximate, ±30%.]

⚠ **Downsampling interacts badly with sparsity.** Our frames are 15% occupied with counts of
1–27. Naïve bilinear downsampling of a sparse count image blurs isolated events into fractional
values and changes the statistic the network sees. If you downsample, **sum-pool (which conserves
event counts) rather than average or interpolate** — or better, re-accumulate the events into a
coarser grid at extraction time, which is exact.

### 4.6 Verdict on ResNet-18

| | |
|---|---|
| Overall | 🟡 **The only one of the three worth an experiment**, in the truncated form of §4.4. |
| Best realistic outcome | [INFER, medium confidence] RMSE roughly comparable to the existing 15 k-parameter CNN — call it **0.50–0.60 N** on a dataset of similar size and composition — with a plausible small gain from better-conditioned low-level features and faster convergence [LIT: He et al. 2019 — pretraining buys convergence speed more reliably than final accuracy]. |
| Worst realistic outcome | Indistinguishable from the from-scratch CNN, at 10× the implementation complexity, with the difference buried under the ±0.09 R² seed noise (§0.5). |
| On the slope-0.576 defect | **Neutral** as a fine-tuned model; **worse** as a frozen probe with ridge (§2.4). |
| Blocking issues | The torchvision version mismatch (§6) must be resolved first. |

---

## 5. Candidate 3 — VGG-16

### 5.1 What it is

VGG-16 (Simonyan & Zisserman, 2015) is the "stack 3×3 convolutions and halve the resolution five
times" architecture. Exactly, from `torchvision/models/vgg.py` config `'D'`:

```
[64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M]
```
where each number is a 3×3 conv (stride 1, pad 1) with that many filters followed by ReLU, and
`M` is a 2×2 max-pool with stride 2. Then `AdaptiveAvgPool2d((7,7))`, flatten to 25,088, and
`Linear(25088→4096) → ReLU → Dropout → Linear(4096→4096) → ReLU → Dropout → Linear(4096→1000)`.

⚠ **No batch normalisation** in the plain `vgg16` (there is a `vgg16_bn` variant). This matters:
BatchNorm is a large part of why ResNets fine-tune stably on small datasets.

### 5.2 Where its parameters are

| part | parameters | share |
|---|---|---|
| block 1 (64 ch) | 38,592 | 0.03% |
| block 2 (128 ch) | 221,184 | 0.16% |
| block 3 (256 ch) | 1,474,560 | 1.07% |
| block 4 (512 ch) | 5,898,240 | 4.26% |
| block 5 (512 ch) | 7,077,888 | 5.12% |
| **conv total (`features`)** | **14,714,688** | **10.6%** |
| `Linear(25088→4096)` | 102,764,544 | **74.3%** |
| `Linear(4096→4096)` | 16,781,312 | 12.1% |
| `Linear(4096→1000)` | 4,097,000 | 3.0% |
| **total** | **138,357,544** | 100% |

(Arithmetic reconciles exactly with the published totals, so these are reliable.)

**89.4% of VGG-16 is a fully-connected head that we must delete anyway** — it requires a fixed
7×7 input, it destroys spatial resolution, and it maps to 1,000 ImageNet classes. What remains is
a 14.7 M-parameter conv stack: still 21× the truncated ResNet-18 of §4.4, and 980× the existing
from-scratch CNN.

### 5.3 The one genuine argument in VGG-16's favour

⚠ **It has direct precedent in exactly this application.** The GelSight force/torque work uses a
CNN "adjusted from VGG-16 networks, pre-trained on ImageNet", with the last fully-connected layer
replaced by an output layer for Fx, Fy, Fz, Tz; **the input is a three-channel difference image
between the current tactile image and the initial reference image**, trained with MSE for
regression [LIT: Yuan, Dong & Adelson, *Sensors* 17(12):2762, 2017]. That is structurally the
same setup as ours: vision-based tactile sensor, difference-against-reference input, regression
to force. It would be dishonest not to lead with this.

Two other credentials worth noting:
- VGG-16's receptive field at `conv5_3` is **196 px = 15.6 mm** (derivation in §2.1), noticeably
  more local than ResNet-18's 435 px, which is arguably better matched to the ~8 mm half-width
  deformation field.
- The `AdaptiveAvgPool2d((7,7))` + flatten path is *not* translation-invariant; it retains a 7×7
  spatial grid. So VGG needs *less* surgery than ResNet on the §2.1 problem.

### 5.4 Why the precedent nonetheless does not carry over

| GelSight setup [LIT] | our setup [REPO] |
|---|---|
| dense RGB photograph of a gel surface under structured LEDs — a genuinely natural-image-like input with smooth shading gradients | 1-channel sparse integer count image, 15% occupancy, max value 27, of a dot lattice under 25 Hz strobing |
| training sets in the thousands of frames, and frames within a press are *not* pure replicates (contact evolves) | 648 independent presses; 50 dwell frames per press are replicates (§14.1a) |
| forces are the applied load on a small, uniform, backed gel | `F = k(x,y)·δ` with `k` varying **9.4×** across an unbacked, edge-strapped sheet (§14.5b, §16.3) |
| 2017-era practice: VGG was near state of the art | 2026: ResNet-18 matches or beats VGG-16 on ImageNet with 12× fewer parameters and 9× fewer FLOPs |

**Cost, concretely.** VGG-16 is ~15.5 GFLOPs at 224×224. At 480×640 that scales to ≈95 GFLOPs
per forward pass, ≈285 GFLOPs forward+backward. At 648 images/epoch that is ~185 TFLOPs/epoch;
at an optimistic 8 effective TFLOP/s fp32 on a 2080 Ti, **≈23 s/epoch**. A full protocol of
300 epochs × 4 folds × 5 seeds = 6,000 epochs is then **~38 hours** (perhaps ~8 h with mixed
precision on the Turing tensor cores). The existing from-scratch CNN completes a whole 4-fold run
in **~30 s** (§15.2g). [INFER; FLOP scaling and throughput assumptions stated, ±2× uncertainty.]

Activation memory is also a real constraint: block 1 alone stores 2 × 64 × 307,200 = 39.3 M
floats = 157 MB per image at native resolution, so a batch of 32 needs ~5 GB for block 1 before
anything else. Add 138 M parameters plus AdamW's two moment buffers (138 M × 16 bytes = 2.2 GB)
and 11 GB gets tight. Workable at batch 8, or at 224×224 — but at 224×224 you have thrown away
the spatial resolution that is the entire signal.

### 5.5 Verdict on VGG-16

❌ **Wrong for this problem. I would not build it.**

It costs 12× ResNet-18's parameters and ~9× its compute; 89% of it is a head you delete; it lacks
BatchNorm, which is precisely the thing that makes small-data fine-tuning behave; and its only
advantage over ResNet-18 (a more local receptive field, and a spatially-aware final pool) can be
obtained from ResNet-18 for free by truncating at `layer2` (§4.4), which gives an even better
7.9 mm receptive field at 0.68 M parameters.

**If the operator wants the VGG lineage tested anyway** — which is a reasonable thing to want,
given the GelSight precedent — the cheap version is: `vgg16(pretrained=True).features[:17]`
(blocks 1–3, 1.73 M params, receptive field 44 px), frozen, as a feature extractor with 3×4
pooling and a ridge head. That costs an afternoon, tests the transfer hypothesis, and does not
commit to 138 M parameters. **Full VGG-16 fine-tuning on 648 samples is not defensible.**

---

## 6. ⚠ Environment blockers to resolve before any of this [ENV]

Checked by inspecting `/home/skymario/.local/lib/python3.8/site-packages/`:

| package | installed | required for this work | issue |
|---|---|---|---|
| `torch` | **2.4.1** (+cu121) | — | fine, CUDA verified working (§14.7) |
| `torchvision` | **0.9.2+cu111** | 0.19.1 for torch 2.4.1 | ⚠⚠ **Mismatched.** torchvision 0.9.2 was built against torch 1.8.1; its compiled `_C` extension is ABI-bound to that version, so `import torchvision` will very likely fail with an undefined-symbol error under torch 2.4.1. **This is the first thing to check** — it is a five-second test and it gates everything. |
| `torchaudio` | 0.8.2 | — | also stale; same ABI issue, harmless if unused |
| `scikit-learn` | **absent** | ridge/PCA/GP | §14.7 already flags this. `pip install scikit-learn==1.3.2` (last release supporting Python 3.8). Alternatively the repo already does closed-form fits with `np.linalg.lstsq` (`train_cnn.py:75`). |
| `timm` | absent | only if you want modern backbones | ⚠ Check its Python-3.8 support before relying on it — I did not verify this. |
| pretrained weights | **no torch hub cache** at `~/.cache/torch/hub/checkpoints/` | — | `download.pytorch.org` resolves from this machine, so the download should work. Expect ~45 MB for `resnet18` and **~528 MB for `vgg16`**. |

**Fix:** `pip install torchvision==0.19.1` — I verified on PyPI that 0.19.1 declares
`requires_python >= 3.8` and ships a `cp38` wheel, so Python 3.8 is supported. ⚠ Pin it, and
consider doing this in a venv rather than `--user`, so a broken install cannot take out the
working `torch` that the rest of the pipeline depends on.

⚠ **API note:** in torchvision 0.9.2 the call is `resnet18(pretrained=True)`; from 0.13 it is
`resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`. Any snippet found online may use either.

---

## 7. What else exists — the survey the brief asks for

### 7.1 ⭐ The physics-factorised model (my actual recommendation)

Not a pretrained network, but it is the option that best fits the evidence, and a fair analysis
has to say so.

**The model:**
```
F̂  =  k̂(x, y)  ·  δ(image)   [ + small correction terms ]
```
- `δ(image)` — deformation magnitude from `marker_features.py`. Force was measured **linear** in
  this, R² = 0.9193, RMSE 0.0923 N at a fixed location, with a `δ^1.5` term adding < 1e-4 R²
  (§15.3). This factor is genuinely translation-invariant, so GAP-style invariance is *correct*
  for it.
- `k̂(x,y)` — from the **measured** stiffness map produced by `franka/map_offset.py`, which is
  already written, deployed and md5-verified (§16.4b), with repeatability 0.042 mm and R²
  0.994–0.9997 on the reference points. `k` is a *calibration*, not something to learn from 648
  samples.
- `(x,y)` at inference from the contact centroid of the difference image (`cx_mm`,
  `extract_features.py`), not from the robot.

**Why this is attractive:**
- **2–3 free parameters.** Cannot overfit. Immune to the p/n problem entirely.
- **Immune to the slope-0.576 failure**: there is no shrinkage estimator in the loop, and no
  label-imbalance-weighted loss.
- **Extrapolates**, which is exactly what fold 4 needs — a stiff edge is handled because `k` was
  *measured* there, not inferred from neighbouring presses.
- Interpretable, and every term is separately falsifiable against physics.

**Honest caveats:** it assumes `F` is separable into `k(x,y)·δ`, which is a leading-order
approximation; §15.3's linearity was verified at **one location over 0–1.1 N** with a 0.092 N
residual that is ~5× the F/T per-frame noise floor, so there *is* unmodelled structure (most
likely field shape / contact geometry). And `map_offset.py`'s 5×5 grid was measured to
under-resolve the surface (residual 0.164 mm vs 0.042 mm repeatability, §16.4b), so the stiffness
map needs the recommended 7×17 density. **Both of those are measurable, not speculative.**

### 7.2 In-domain pretrained models that actually exist

| model | what it is | fit here |
|---|---|---|
| **Sparsh** (Meta, CoRL 2024) [LIT] | Self-supervised (MAE / DINO / I-JEPA) ViT encoders pretrained on **460 k+ tactile images** from DIGIT, GelSight'17 and GelSight Mini. Benchmarked on TacBench, which includes force estimation; reported to outperform task- and sensor-specific end-to-end training by ~95% on average across TacBench. | 🟡 **The closest thing to in-domain tactile pretraining that exists** — and it is genuinely tactile, which is more than ImageNet can say. ⚠ But it is trained on **dense RGB gel images**, not sparse single-polarity event counts of a marker lattice. The sensor modality mismatch is arguably as large as ImageNet's, just differently shaped. Also ViT-based (needs 3-channel, ~224×224 patched input) and Python-3.8 support is unverified. **Worth 2 hours to test as a frozen feature extractor; not worth building a project around.** |
| **Masked Event Modeling / MEM** (Klenk et al., WACV 2024) [LIT] | ViT pretrained self-supervised on **event histograms** (640×480 resized to 512×512), weights released (`github.com/tum-vision/mem`). Reports SOTA on N-ImageNet / N-Cars / N-Caltech101, and explicitly that event-domain pretraining beats supervised RGB pretraining on real event data. | 🟡 **The only "pretrained on event data" option with released weights.** ⚠ But it is pretrained on *moving natural scenes*, where events trace object contours. Our events come from a static strobed marker lattice. The statistics are not the same. Still, it is the one candidate whose *input contract* (an event histogram) matches ours natively. |
| **TESPEC** (ICCV 2025) [LIT] | Temporally-enhanced self-supervised pretraining for event cameras. | 🟡 Same caveat as MEM; newer, less validated externally. Worth a look if MEM shows promise. |
| **Evetac** (Funk et al., IEEE T-RO 2024) [LIT] | An **event-based optical tactile sensor** — the closest published system to E-BTS. Tracks elastomer markers from sparse events at 1000 Hz; **normal displacement is predicted by a CNN trained on synchronised force–displacement–event data**, shear from marker tracking. | ⭐ **Read this paper.** Not a pretrained model you can download, but it is the direct methodological precedent, and it independently arrives at the same split we did: **marker tracking for shear, a learned model for normal.** |
| **Dense force estimation from an event tactile sensor** (2026) [LIT] | U-Net, **7.9 M parameters**, input a 2×90×112 polarity-separated Surface-of-Active-Events tensor, **trained from scratch — no pretrained backbone**, on 219 recordings (~17 min of events). Reported MAE: shear 0.10–0.14 N, **normal 0.93 ± 1.00 N** over a 0–20 N range. | ⚠ **A very useful calibration point.** A dedicated, recent, event-tactile force network with 500× our current model's parameters, trained from scratch, achieves 0.93 N MAE on 0–20 N. Our 0.509 N RMSE on 0.24–5.95 N is not obviously behind that. **This is evidence that (a) from-scratch is normal in this niche, and (b) our numbers are not embarrassing.** |

### 7.3 General-purpose alternatives

- **DINOv2** (self-supervised ViT, Meta 2023): known for unusually strong *frozen* linear probes,
  which is exactly the regime we are in. ⚠ Requires 3-channel, patch-14, ≥224 input; same
  natural-image domain gap; ViT patch tokens do retain position, which helps on §2.1. Cheap to
  test as a frozen probe (option 1 in §4.4's ladder), expensive to justify beyond that.
- **Small modern CNNs**: MobileNetV3-Small (~2.5 M), EfficientNet-B0 (~5.3 M), ConvNeXt-Tiny
  (~28 M). All still 100×+ the from-scratch CNN. If you want a pretrained backbone that is
  actually sized for 648 samples, **MobileNetV3-Small truncated early** is a better choice than
  either ResNet-18 or VGG-16 [INFER].
- **Gaussian process regression** on ~10–20 marker features: natural at n ≈ 650, gives calibrated
  *uncertainty* (which the application probably wants), and its posterior mean does not suffer
  the arbitrary shrinkage of a hand-tuned ridge penalty. Available via scikit-learn once
  installed. ⚠ Needs a sensible kernel and a mean function; a zero mean will reintroduce
  regression-to-the-mean at the edges.
- **Gradient boosting** (§14.8 step 4): strong in the few-hundred-to-few-thousand-sample regime,
  gives feature importances. `xgboost` not installed; sklearn's `HistGradientBoostingRegressor`
  would do.
- **Just keep the 15 k-parameter from-scratch CNN.** It is correctly sized, it already handles
  the translation-invariance problem properly, and He et al. (2019) [LIT] is direct evidence that
  from-scratch training matches pretraining given enough iterations. ⚠ **"We evaluated pretrained
  backbones and they did not beat a correctly-sized from-scratch model" is a legitimate,
  publishable result** — and given §0.5's noise-floor arithmetic, it is a likely one.

---

## 8. Comparison table

| | **TDNN** | **ResNet-18** | **VGG-16** | *(ref) existing CNN* | *(ref) physics-factorised* |
|---|---|---|---|---|---|
| **Input required** | sequence `[T, d]` of per-frame feature vectors; T ≈ 90, d ≈ 10–40 | `[B, 3, H, W]` float, ImageNet-normalised | `[B, 3, H, W]` float, ImageNet-normalised | `[B, 3, 120, 160]` (diff + 2 coord channels) | ~10 scalar marker features + (x,y) |
| **Pretrained weights exist for our modality?** | ❌ **No** — only speech (80-d mel filterbanks, 100 fps) | 🟡 ImageNet only | 🟡 ImageNet only | n/a | n/a |
| **Parameters** | 20–100 k (from scratch) | 11.7 M (0.68 M truncated at layer2) | 138 M (14.7 M conv only) | ~15 k | ~2–3 |
| **params / independent sample (n=648)** | ~50–150 | 18,040 (1,051 truncated) | 213,515 | 23 | 0.005 |
| **Translation invariance** | invariant *in time*, which is fine — time is not the confound | ⚠ GAP is fully invariant. Must be replaced (3×4 pool + coord channels + truncation) | 🟡 7×7 pool + flatten retains coarse position; less surgery needed | ✅ already solved (3×4 pool + coord channels) | ✅ solved by construction — invariance applies only to the δ factor |
| **Final receptive field** | ~0.6–1.2 s of time | 435 px = 34.5 mm (layer4) / 99 px = 7.9 mm (layer2) | 196 px = 15.6 mm | ~n/a (small net) | n/a |
| **Domain gap** | total — no shared input structure with speech | large; conv1 partly transfers, layer3–4 essentially not | same, plus 89% of the model is a discarded head | none | none |
| **Effect on slope-0.576** | 🟡 **could help** — turns 648 scalars into 648 trajectories, attacking label imbalance | neutral (fine-tuned) / ⚠ worse (ridge probe) | ⚠ mildly worse (more regularisation needed) | the baseline defect | ✅ immune |
| **Compute / epoch, 648 samples, 2080 Ti** | seconds | ~2–5 s | ~23 s (fp32, 480×640) | ~0.1 s | milliseconds (CPU) |
| **Blocking issues** | no pretrained target exists | torchvision ABI mismatch (§6) | same, plus 528 MB weights and 38 h protocol | none | needs the 7×17 stiffness map |
| **Verdict** | ❌ as pretrained; 🟡 as a from-scratch secondary experiment | 🟡 **the one worth trying**, truncated | ❌ **do not build** | ✅ keep as the reference | ⭐ **build this first** |

---

## 9. Recommendation

**Ordered, with the reasoning attached to each step.**

**Step 0 — five minutes, before anything else.** Check whether `import torchvision` works under
`torch 2.4.1`. It probably does not (§6). If not, `pip install torchvision==0.19.1` in a venv.
Everything below is blocked on this.

**Step 1 — build the physics-factorised model (§7.1).** `k̂(x,y)` from `map_offset.py` × `δ` from
`marker_features.py`. Two or three parameters. This should be the **primary baseline** that every
subsequent model is measured against, alongside predict-the-mean and depth-only. Reasoning: it
encodes the measured physics (`F = k·δ`, linear, R² = 0.919, §15.3), it cannot overfit, it
extrapolates to the stiff edge where every learned model failed, and it is immune to the
compression defect. If it works, the ML question becomes "what does a network add on top of
known physics?", which is a much better-posed question than "can a network learn the physics from
648 samples?".

**Step 2 — the cheapest possible falsification of the transfer hypothesis.** Frozen
ResNet-18 + GAP → 512 features → ridge, strict row-band CV. Seconds of compute. Then the same
with 3×4 pooling and with a `vgg16.features[:17]` truncation for comparison. **If frozen ImageNet
features cannot beat predict-the-mean, the whole pretrained-backbone direction is answered and
you have spent an afternoon.** Report it either way — a clean negative is a result.

**Step 3 — only if step 2 shows signal:** ResNet-18 truncated at `layer2`, conv1 summed to 1
channel + 2 zero-initialised coordinate channels, 3×4 pooling, LP-FT (§2.3d), 5 seeds, strict
grouped CV, reporting RMSE **and slope** (§2.4).

**Step 4 — do not build VGG-16.** §5.5.

**Step 5 — TDNN, if at all, later and reframed.** Not as a pretrained model (none exists for this
modality — this needs to be said to the operator explicitly, since §16.4a records the decision
under the assumption that it does). As a from-scratch temporal model over the dip ramp, for one
specific purpose: converting 648 scalar labels into 648 force trajectories, which attacks the
label-imbalance half of the compression defect. Build the spatial model first.

**Step 6 — the thing that will actually move the number.** §16.4 Phase 0 and Phase 3: re-mount or
formally restrict the strapped edge, and **design the next collection by target force rather than
by depth**, so the force histogram is balanced rather than 9%-above-3 N. Per §0.5, an
architecture difference smaller than ΔR² ≈ 0.16 is not measurable at n = 648. **Data design is
above the noise floor; architecture selection is below it.**

⚠ **One honest caveat on this whole document.** The dataset it reasons about has been deleted
(§16.3), and the next one will differ in force distribution, possibly in mounting, and possibly
in indenter. Every quantitative prediction here should be re-checked against the new data rather
than inherited. What should survive unchanged are the *structural* arguments: the parameter
distribution inside ResNet-18 and VGG-16, the receptive-field arithmetic, the non-existence of a
relevant pretrained TDNN, the mathematics of why slope < 1 means over-shrinkage, and the
noise-floor arithmetic in §0.5.

---

## 10. Glossary

| term | meaning |
|---|---|
| **backbone** | the feature-extracting body of a network, without its task-specific output head |
| **BatchNorm** | normalises each channel by batch statistics; stabilises training and permits larger learning rates |
| **dilation** | a convolution whose kernel skips input positions, widening its receptive field without extra parameters |
| **fine-tuning** | continuing to train pretrained weights on your data |
| **GAP (global average pooling)** | averaging each channel's whole feature map to one number; makes the output translation-invariant |
| **grouped CV** | cross-validation where all samples from a group (here: a grid row-band) are in the same fold, so neighbours cannot leak between train and test |
| **linear probe** | freezing the backbone, extracting features once, and fitting only a linear model on top |
| **LP-FT** | linear-probe first, then fine-tune — reduces the feature distortion that naïve fine-tuning causes |
| **p/n** | number of features ÷ number of samples; > 1 means an unregularised linear fit is under-determined |
| **receptive field** | the region of the input that one unit of a given layer can see |
| **ridge regression** | least squares plus an L2 penalty on the coefficients; a *shrinkage* estimator |
| **shrinkage** | pulling estimates toward zero (or the mean) to trade bias for variance |
| **TDNN** | time-delay neural network: a 1-D convolution over the time axis with weights shared across time |
| **transfer learning** | reusing weights learned on one task as the starting point for another |

---

## 11. Sources

**Transfer learning, general**
- Yosinski, Clune, Bengio & Lipson (2014), *How transferable are features in deep neural
  networks?*, NeurIPS — <https://arxiv.org/abs/1411.1792>
- Raghu, Zhang, Kleinberg & Bengio (2019), *Transfusion: Understanding Transfer Learning for
  Medical Imaging*, NeurIPS — <https://arxiv.org/abs/1902.07208>
- Kornblith, Shlens & Le (2019), *Do Better ImageNet Models Transfer Better?*, CVPR —
  <https://arxiv.org/abs/1805.08974>
- He, Girshick & Dollár (2019), *Rethinking ImageNet Pre-training*, ICCV —
  <https://arxiv.org/abs/1811.08883>
- Kumar, Raghunathan, Jones, Ma & Liang (2022), *Fine-Tuning can Distort Pretrained Features and
  Underperform Out-of-Distribution*, ICLR — <https://arxiv.org/abs/2202.10054>
- Geirhos et al. (2019), *ImageNet-trained CNNs are biased towards texture*, ICLR —
  <https://arxiv.org/abs/1811.12231>

**Architecture / spatial encoding**
- Liu et al. (2018), *An Intriguing Failing of Convolutional Neural Networks and the CoordConv
  Solution*, NeurIPS — <https://arxiv.org/abs/1807.03247>
- Islam, Jia & Bruce (2020), *How much Position Information Do Convolutional Neural Networks
  Encode?*, ICLR — <https://arxiv.org/abs/2001.08248>
- torchvision model definitions, read locally at
  `/home/skymario/.local/lib/python3.8/site-packages/torchvision/models/{resnet,vgg}.py`

**TDNN**
- Waibel, Hanazawa, Hinton, Shikano & Lang (1989), *Phoneme recognition using time-delay neural
  networks*, IEEE TASSP 37(3)
- Peddinti, Povey & Khudanpur (2015), *A time delay neural network architecture for efficient
  modeling of long temporal contexts*, Interspeech
- Snyder et al. (2018), *X-vectors: Robust DNN embeddings for speaker recognition*, ICASSP
- Desplanques, Thienpondt & Demuynck (2020), *ECAPA-TDNN*, Interspeech —
  <https://arxiv.org/abs/2005.07143>

**Event cameras**
- Gehrig, Loquercio, Derpanis & Scaramuzza (2019), *End-to-End Learning of Representations for
  Asynchronous Event-Based Data*, ICCV — <https://arxiv.org/abs/1904.08245>
- Klenk, Bonello, Koestler, Araslanov & Cremers (2024), *Masked Event Modeling: Self-Supervised
  Pretraining for Event Cameras*, WACV — <https://arxiv.org/abs/2212.10368>, weights at
  <https://github.com/tum-vision/mem>
- Mohammadi et al. (2025), *TESPEC: Temporally-Enhanced Self-Supervised Pretraining for Event
  Cameras*, ICCV — <https://arxiv.org/abs/2508.00913>
- Gallego et al. (2020), *Event-based Vision: A Survey*, IEEE TPAMI —
  <https://arxiv.org/abs/1904.08405>

**Tactile sensing**
- Yuan, Dong & Adelson (2017), *GelSight: High-Resolution Robot Tactile Sensors for Estimating
  Geometry and Force*, Sensors 17(12):2762 — <https://www.mdpi.com/1424-8220/17/12/2762>
  (the VGG-16-on-difference-image force regressor)
- Funk, Helmut, Chalvatzaki, Calandra & Peters (2024), *Evetac: An Event-based Optical Tactile
  Sensor for Robotic Manipulation*, IEEE T-RO — <https://arxiv.org/abs/2312.01236>
- *Dense Force Estimation with an Event-based Optical Tactile Sensor* (2026) —
  <https://arxiv.org/abs/2606.09451> (U-Net, 7.9 M params, from scratch, 219 recordings, normal
  force MAE 0.93 ± 1.00 N over 0–20 N)
- Higuera, Sharma et al. (2024), *Sparsh: Self-supervised touch representations for vision-based
  tactile sensing*, CoRL — <https://arxiv.org/abs/2410.24090>, code/weights at
  <https://github.com/facebookresearch/sparsh>

**Imbalanced regression (the slope-0.576 problem)**
- Yang, Zha, Chen, Wang & Katabi (2021), *Delving into Deep Imbalanced Regression*, ICML —
  <https://arxiv.org/abs/2102.09554>
- Ren, Zhang, Yu & Liu (2022), *Balanced MSE for Imbalanced Visual Regression*, CVPR Oral —
  <https://arxiv.org/abs/2203.16427>

**This repository**
- `HANDOFF.md` §12.5, §13.1–13.6, §14.1–14.9, §15.2c–15.2g, §15.3, §16.1–16.5
- `ml/README.md`, `ml/train_cnn.py`, `ml/build_dataset.py`, `ml/extract_features.py`,
  `ml/evaluate_holdout.py`
