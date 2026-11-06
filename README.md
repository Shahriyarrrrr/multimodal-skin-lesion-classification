# Multimodal Skin Lesion Classification

Undergraduate thesis (CSE, American International University-Bangladesh). The project tests whether adding patient metadata (age, sex and anatomical site) to a dermoscopy image classifier improves 7-class skin lesion classification. The classifier is EfficientNet-B3 with MetaBlock fusion, trained on HAM10000, and the work also covers:

- **XAI validation:** Grad-CAM heatmaps scored against expert lesion masks.
- **External validation:** testing on BCN20000, a dataset from a different hospital.
- **Demographic disaggregation:** results broken down by age, sex and anatomical site.

Every phase runs as a Jupyter notebook. Each phase verifies the outputs of the previous one (file hashes, row counts, read-back checks) before it computes anything new. Comparison rules were written down and hash-locked before any results existed.

| | |
|---|---|
| **Training data** | HAM10000: 10,015 images, 7 classes (akiec, bcc, bkl, df, mel, nv, vasc) |
| **External test data** | BCN20000: 16,284 images (4,591 lesions) in the 7 HAM10000 classes |
| **Main model** | EfficientNet-B3 + MetaBlock (Pacheco & Krohling, 2021) |
| **Comparisons** | Matched image-only ablation, ResNet50, ViT-B/16 |
| **Splits** | Lesion-grouped: ~15% internal holdout + 5-fold CV, no lesion appears in two splits |
| **Statistics** | DeLong test + lesion-level bootstrap 95% CI, Holm correction |
| **Hardware** | Single NVIDIA RTX 5060 Ti, Windows 11, bf16 mixed precision |

---

## Repository structure

```
Phase - 1/
  Phase1_DataSetup-P1.ipynb        early data setup and hair-removal prototyping
  P2/Phase1_DataSetup-P2.ipynb     final Phase 1: verification, preprocessing, splits, class balancing
Phase - 2/
  P1/Phase2_P1.ipynb               baseline training, first iteration
  P2/Phase 2_P2.ipynb              baseline training, second iteration
  P3/Phase 2_P3.ipynb              final baseline run + gate check (locked reference values)
Phase - 3/
  P1/Phase 3_P1.ipynb              multimodal model + matched ablation, 5-fold CV, model freeze
Phase - 4/
  P1/Phase 4_P1.ipynb              Grad-CAM XAI validation against lesion masks
Phase - 5/
  P1/Phase 5_P1.ipynb              holdout metrics, BCN20000 external validation, demographic tables
Phase - 6/
  Phase 6_P1.ipynb                 statistical testing, confidence intervals, results digest
```

Each code cell opens with a docstring naming it, for example `05_01_multiscale_blackhat`, and saying what it needs and what it writes. Run the cells in order, top to bottom.

---

## Pipeline

### Phase 1: Data setup and preprocessing

- **Verification.** Checks the image count and metadata row count (10,015 each), corrupted files, and missing or extra images.
- **Segmentation masks.** Uses the Tschandl HAM10000 lesion segmentations. The ISIC 2018 Task 1 masks were rejected after a programmatic check found 0% overlap with HAM10000 image IDs.
- **Hair removal.** Hair is detected with multi-scale BlackHat (kernels 9 and 17), then blur plus adaptive-OR-Otsu thresholding, then morphological cleaning. Two filters follow: one excludes the lesion interior, and one keeps only thin connected components (extent ≤ 0.25). The detected hair is removed with Telea inpainting.
- **Contrast.** CLAHE (clip 1.5, 8×8 tiles) on the L channel in LAB colour space only, so lesion hue is preserved.
- **Crop and resize.** A lesion-aware adaptive crop or pad: a 450×450 base crop, widened to the lesion bounding box plus a 10 px margin when needed, with reflect padding. The result is resized to 300×300, using INTER_AREA for images and nearest-neighbour for masks.
- **Normalisation and augmentation.** Both are applied at training time only. Normalisation uses ImageNet statistics. Augmentation uses horizontal and vertical flips, rotation up to ±180°, zoom ±10% and brightness ×0.8–1.2.
- **Splits.** `StratifiedGroupKFold`, grouped by `lesion_id` and stratified by diagnosis × site:
  - 15% holdout: 1,432 images from 1,067 lesions
  - 5-fold CV: 8,583 images
- **Class imbalance.** Two independent mechanisms are used together:
  - Effective-number class weights (Cui et al., 2019, β = 0.999), computed from each fold's own training counts
  - A balanced `WeightedRandomSampler`
- **Reproducibility.** Seed 42 everywhere, deterministic cuDNN, and a saved environment fingerprint.

### Phase 2: Baselines

Three models are trained with 5-fold CV. All use AdamW (weight decay 1e-5), batch size 32, up to 100 epochs, best-epoch selection by validation AUC-ROC, bf16 autocast and resumable checkpoints.

| Model | Input | Class balancing | Head |
|---|---|---|---|
| EfficientNet-B3 (image-only) | 300×300 | Yes | Dropout(0.5) → Linear |
| ResNet50 (ImageNet1K_V1) | 300×300 | No | Linear |
| ViT-B/16 | 224×224 | No | Linear |

### Phase 3: Multimodal model

```
image (3×300×300)  → EfficientNet-B3 features ──────────→ feature map (1536×10×10) ─┐
                                                                                     ├→ MetaBlock → Global Avg Pool → Dropout(0.5) → Linear(7)
metadata (20)      → Dense(32) → ReLU → Dropout(0.3) ──→ metadata features (32) ────┘
```

- **Metadata vector (20 values).**
  - Age, z-scored with the training fold's own mean and standard deviation
  - An age-recorded flag
  - Sex, one-hot (3 values)
  - Anatomical site, one-hot (15 values)
- **MetaBlock.** `sigmoid(tanh(f(m) ⊙ F) + g(m))`, applied to the *unpooled* feature map. Pooling happens after the fusion, following the paper's equations.
- **Matched ablation.** Identical to the multimodal model except that the metadata branch and MetaBlock are removed. Both models use the same backbone, head, initialisation and training recipe.
- **Two-phase training.**
  - Epochs 1–5: the backbone is frozen, including its BatchNorm statistics.
  - Epoch 6 onwards: the whole network trains, with a backbone learning rate of 1e-4 and a head learning rate of 1e-3.
  - Early stopping uses patience 10, counted from epoch 6.
- **Model freeze.** The fold with the best validation AUC-ROC (fold 3) is copied, protected with a SHA-256 hash, reloaded to confirm it reproduces its scores, and documented in a model card. No later phase changes it.

### Phase 4: XAI validation

- Grad-CAM is run on all 1,432 holdout images for the frozen multimodal model and for the fold-3 ablation.
- Two layers are used: the final layer `backbone[-1]` (primary) and `backbone[5]` (secondary).
- Each heatmap is binarised two ways (Otsu threshold, and the top 50% of its peak) and scored with Dice and IoU against the lesion masks.
- The rule for "comparable" was recorded and hash-locked before any score was computed. Comparisons are paired per image, with lesion-level bootstrap intervals.

### Phase 5: External validation and demographics

- **Holdout.** The full metric set on the 1,432-image holdout: macro AUC-ROC, balanced accuracy, macro-F1, macro precision, macro PR-AUC, ECE (15 bins), Brier score and accuracy.
- **BCN20000 labels and sites.** BCN20000 labels and sites are mapped onto the HAM10000 classes with closed dictionaries, so any unexpected value raises an error.
  - Squamous cell carcinoma (SCC) is kept as a secondary observation.
  - Melanoma metastasis, scars and undiagnosed images are excluded.
- **BCN20000 preprocessing ("Recipe A").** CLAHE, then a resize from 1024 to 300 pixels. There is no hair removal and no cropping.
- **BCN20000 metadata.** Encoded with the training fold's age statistics. A site-blind variant (site set to unknown for every image) is also run.
- **Generalisation gap.** The holdout-to-BCN20000 drop is compared with the published figures from Combalia et al. (2022).
- **Demographic tables.** Results by age group (0–20, 21–40, 41–60, 61–79, 80+, unknown), by sex and by site. A two-tier rule applies: any cell with fewer than 30 positives or negatives is flagged as low sample instead of being given an estimate.

### Phase 6: Statistics

- **Rules note.** A hash-locked rules note is written before any result exists.
- **Null gate.** A shuffled-label check must pass before the main comparison runs.
- **Claim 1.** Multimodal vs ablation, per class, on pooled out-of-fold predictions. A difference needs both a Holm-corrected DeLong test and a lesion-bootstrap 95% CI.
- **Secondary comparisons.** Multimodal vs ResNet50 and vs ViT-B/16.
- **Intervals.** Lesion-bootstrap intervals for every holdout, BCN20000 and demographic figure.
- **Exit gate.** A results digest and an exit gate that re-hashes all 71 frozen inputs.

---

## Results

Results labelled **EXPLORATORY** came from analyses chosen after earlier results had been seen. They are reported next to the primary results, never in place of them. All statements about metadata apply to this setup only: three metadata fields, MetaBlock fusion and HAM10000.

### Cross-validation (Phase 3, validation folds)

| Model | Macro AUC-ROC | Macro-F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| Multimodal | 0.9355 ± 0.0102 | 0.6529 ± 0.0437 | 0.6926 ± 0.0388 | 0.9528 ± 0.0071 |
| Image-only ablation | 0.9234 ± 0.0403 | 0.5604 ± 0.1536 | 0.6540 ± 0.0922 | 0.9427 ± 0.0226 |

### Claim 1: Does metadata fusion help? (pooled out-of-fold, 8,583 images)

| Class | N | AUC multimodal | AUC ablation | Δ [95% CI] | Holm p | Reading |
|---|---|---|---|---|---|---|
| AKIEC | 281 | 0.8887 | 0.9127 | −0.0241 [−0.0478, −0.0011] | 0.0841 | not detectable* |
| BCC | 438 | 0.9548 | 0.9482 | +0.0067 [−0.0080, +0.0216] | 0.7074 | not detectable |
| BKL | 941 | 0.8967 | 0.8882 | +0.0085 [−0.0046, +0.0217] | 0.7074 | not detectable |
| DF | 100 | 0.9539 | 0.9040 | +0.0499 [+0.0134, +0.0876] | 0.0285 | multimodal higher |
| MEL | 954 | 0.8514 | 0.8417 | +0.0097 [−0.0067, +0.0248] | 0.7074 | not detectable |
| NV | 5746 | 0.9307 | 0.9188 | +0.0120 [+0.0051, +0.0186] | 2.2e-4 | multimodal higher |
| VASC | 123 | 0.9764 | 0.9659 | +0.0104 [−0.0269, +0.0448] | 0.7074 | not detectable |

\*The DeLong test and the bootstrap disagree for AKIEC, so it is counted as not detectable.

On pooled out-of-fold predictions, metadata fusion was associated with higher AUC-ROC in 2 of 7 classes (DF, NV) and showed no detectable difference in the rest. This result did not hold in two exploratory checks:

- With fold 2 removed, the ablation was higher in AKIEC and no class favoured the multimodal model.
- On the internal holdout, the ablation was higher in AKIEC and DF.

### Holdout and BCN20000 (one fold-3 model per arm)

| Metric | Holdout: multimodal | Holdout: ablation | BCN20000: multimodal | BCN20000: ablation |
|---|---|---|---|---|
| Macro AUC-ROC | 0.9348 | 0.9595 | 0.7558 | 0.7894 |
| Balanced accuracy | 0.6649 | 0.7293 | 0.3468 | 0.3801 |
| Macro-F1 | 0.6221 | 0.6917 | 0.2909 | 0.3418 |
| Accuracy | 0.7640 | 0.7849 | 0.4393 | 0.4764 |

The macro AUC-ROC gap from the holdout to BCN20000 was 0.179 for the multimodal model and 0.170 for the ablation. The best ISIC 2019 algorithm dropped 0.075 (Combalia et al., 2022), but that is the closest published figure, not a like-for-like comparison.

### Claim 2: XAI (Grad-CAM vs lesion masks, final layer, Otsu threshold)

| Model | Dice | IoU |
|---|---|---|
| Multimodal | 48.0% | 34.4% |
| Image-only ablation | 62.6% | 47.3% |

The paired Dice difference is −14.62 percentage points [−16.28, −12.92]. With the top-50% threshold, there was no detectable difference between the two models.

### Baselines (Phase 2, 5-fold mean ± SD macro AUC-ROC)

| EfficientNet-B3 (image-only) | ResNet50 | ViT-B/16 |
|---|---|---|
| 0.9316 ± 0.0298 | 0.9608 ± 0.0036 | 0.9609 ± 0.0041 |

ResNet50 and ViT-B/16 each scored higher than the multimodal model in 5 of 7 classes. The comparison does not isolate the effect of metadata, because the baselines used a different training recipe (no class balancing).

### Demographic disaggregation

On the holdout, each model has 168 demographic cells: 22 have estimates, 91 are flagged low sample and 55 are undefined. AKIEC, DF and VASC never reach 30 positives in any holdout subgroup. Site is both a model input and a disaggregation axis, so these results show consistency given site as an input, not unconditional fairness. Skin tone is not addressed.

---

## Reproducing

### 1. Data

The datasets are not included in this repository. Download them from the original sources:

- **HAM10000** images, metadata and Tschandl lesion segmentations: [Harvard Dataverse, doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- **BCN20000**: [ISIC Archive](https://www.isic-archive.com/)

The notebooks expect this layout under a single base folder:

```
<BASE_DIR>/
└── Resources/
    ├── Dataset/
    │   ├── HAM10000/
    │   │   ├── images/
    │   │   └── HAM10000_metadata.tab
    │   ├── HAM10000_Tschandl_Segmentations/
    │   │   └── HAM10000_segmentations_lesion_tschandl/
    │   └── BCN20000/
    │       ├── images/
    │       └── bcn20000_metadata_<date>.csv
    └── Artifacts/        ← created by the notebooks, one folder per phase
```

`BASE_DIR` is hard-coded as `T:\Thesis - Development` in each notebook's configuration cells. Find and replace it with your own path before running. In Phase 5, also update the BCN20000 metadata filename to match your download.

### 2. Environment

The original runs used:

| Component | Version |
|---|---|
| Python | 3.11 (Anaconda) |
| PyTorch | 2.11 + CUDA 12.8 |
| torchvision | 0.26 |
| OpenCV | 5.0 |
| NumPy | 2.4 |
| pandas | 3.0 |
| scikit-learn | 1.9 |

```bash
conda create -n thesis-core python=3.11
conda activate thesis-core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python numpy pandas scikit-learn scipy statsmodels matplotlib pillow tqdm jupyter
```

### 3. Run order

```
Phase 1 (P2) → Phase 2 (P3) → Phase 3 → Phase 4 → Phase 5 → Phase 6
```

Long-running steps checkpoint as they go and resume after an interruption. Each phase ends with an exit gate that must pass before the next phase can start.

---

## Limitations

- Holdout and BCN20000 results come from a single fold-3 model per arm. They do not reflect training variance across folds.
- Pooled out-of-fold results are optimistic, because the best epochs were selected on the same folds.
- The DeLong test ignores lesion clustering. The lesion bootstrap covers the confidence intervals but not the DeLong p-values.
- The BCN20000 gap combines hospital, camera and preprocessing differences, which cannot be separated here.
- Phase 3 predictions are bf16, while Phases 4 and 5 use fp32. No table mixes the two without a label.

---

## References

- Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161.
- Hernández-Pérez, C. et al. (2024). BCN20000: Dermoscopic lesions in the wild. *Scientific Data*, 11, 641.
- Pacheco, A. G. C. & Krohling, R. A. (2021). An attention-based mechanism to combine images and metadata in deep learning models applied to skin cancer classification. *IEEE JBHI*, 25(9).
- Cui, Y. et al. (2019). Class-balanced loss based on effective number of samples. *CVPR*.
- Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics*, 44(3).
- Combalia, M. et al. (2022). Validation of artificial intelligence prediction models for skin cancer diagnosis using dermoscopy images. *The Lancet Digital Health*, 4(5), e330–e339.

---

**Supervisor:** Prof. Dr. Khandaker Tabin Hasan, American International University-Bangladesh
