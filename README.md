# The Learnability Gap in Medical Latent Diffusion

**Accepted at MICCAI 2026** — See you in Strasbourg! 🇫🇷

### 📄 [Read the paper](https://link.springer.com/chapter/10.1007/978-3-032-38189-7_52)

## TL;DR

> **Perfect reconstruction ⇏ perfect latents.**
> Medical autoencoders preserve every discriminative feature a classifier needs, near-losslessly. But they arrange it in a way classifiers can't learn from. We call this the **learnability gap**.

Latent diffusion is a natural fit for fixing class imbalance in medical imaging (e.g. rare findings in chest X-rays): a hospital could train locally and share synthetic samples on demand. But this only works if the pipeline can actually learn what defines the rare classes — and prior work only checks whether the *autoencoder* can reconstruct them, not whether a classifier can learn from the *latents*.

We show it usually can't: across **5 autoencoder families** and **4 medical benchmarks**, classifiers trained directly on latent codes underperform classifiers trained on the reconstructed images or the raw images themselves ($p = 9.5\times10^{-7}$), even though the autoencoder reconstructs the discriminative content near-losslessly. Medical-domain fine-tuning of the autoencoder improves reconstruction fidelity but does **not** close this gap.

![Method overview: a frozen pretrained autoencoder maps images to latent space and back. Reconstruction-space classifiers match image-space performance, but latent-space classifiers degrade substantially. A noise-conditioned ConvNeXt-Tiny student is distilled from an image-space teacher to partially narrow the gap.](assets/modelabstract.png)

*Method overview. A frozen pretrained autoencoder maps images to latent space and back. Reconstruction-space classifiers match image space, but latent-space classifiers degrade substantially. A ConvNeXt-Tiny student classifier with FiLM-based noise conditioning is distilled from an image-space teacher to help narrow the gap.*

## Key takeaways

1. **Check the latent space, not just reconstruction quality.** When evaluating a latent diffusion pipeline, train a classifier on the latents themselves, not only on decoded reconstructions.
2. **Large pretrained autoencoders are already sufficient.** Medical-domain fine-tuning is not needed and does not close the gap (reconstruction fidelity improves, latent learnability does not).
3. **Latent-space restructuring — not reconstruction fidelity — is the open problem** for generating useful synthetic medical data.

## Setup

```bash
git clone https://github.com/MischaD/LearnabilityGap.git && cd LearnabilityGap
pip install torch torchvision diffusers medvae scikit-learn pandas mlxtend ml_collections opencv-python tqdm einops matplotlib
```

Run everything from the repo root with `PYTHONPATH=.`. Provide the four dataset CSVs under `data/`; the `path` column resolves against whatever you pass as `--data_dir`.

- **Autoencoders (5 families, frozen):** SD 1.4 · Flux.1 dev · Flux.2 dev · MedVAE · MedVAE fine-tuned per dataset
- **Classifiers:** ResNet-50 (ImageNet-pretrained) for image space and reconstruction space; ConvNeXt-Tiny for latent space
- **Datasets:**

| Dataset | Modality | Classes | N |
|---|---|---|---|
| MIMIC-CXR | Chest radiography | 19 findings | 111k |
| ISIC-2019 | Dermatoscopy | 8 lesion types | 25k |
| CT-RATE | Computed tomography | 13 findings | 23k |
| Cardium | Echocardiography | CHD vs. normal | 6.6k |

To probe and partially narrow the gap, we also introduce **noise-conditioned latent classifiers** (FiLM-modulated ConvNeXt-Tiny, distilled from an image-space ResNet-50 teacher), which offer **64× throughput** and **120× memory** gains over image-space models while serving as diagnostic tools for latent space quality.

## Running

```bash
# 1. Precompute VAE latents (once per AE)
python scripts/compute_latents.py --filelist data/mimic.csv --basedir /path/to/images --output_latents outputs/latents_flux2

# 2. Image-space baseline (ResNet-50, LDAM+DRW, 5-fold CV)
python scripts/classifier_train.py --data_dir /path/to/images --filelist data/mimic.csv --out_dir outputs/img --loss ldam --rw_method cb --drw --do_crossfold

# 3. Latent-space classifier (ConvNeXt-Tiny on precomputed latents)
python scripts/classifier_train.py --data_dir outputs/latents_flux2 --filelist data/mimic.csv --out_dir outputs/lat --model_name ConvNeXt-Tiny --is_latent --mean_path outputs/latents_flux2/latents_channel_mean.pt --loss ldam --rw_method cb --drw --do_crossfold
```

Swap in `classifier_train_noise_cond.py` for FiLM noise conditioning, `classifier_distill.py` for image-space distillation from a trained teacher, or `classifier_distill_noise_cond.py` for both. See `scripts/run_all_trainings.sh` for the full 5 AEs × 4 datasets sweep behind Table 1.

## Citation

```bibtex
@inproceedings{10.1007/978-3-032-38189-7_52,
	abstract = {Generative data augmentation with latent diffusion models is a promising strategy for addressing class imbalance in medical imaging, yet current approaches focus on perceptual fidelity and domain-specific autoencoder fine-tuning while neglecting a more fundamental bottleneck. We identify and formalize the learnability gap: large-scale pretrained autoencoders faithfully encode discriminative features for medical classification, as evidenced by near-lossless performance in reconstruction space, yet their latent representations are structured in ways that are difficult for classifiers to learn from. Across five autoencoder families and four medical benchmarks spanning chest radiography, dermatoscopy, computed tomography, and echocardiography, we show that this gap persists regardless of architecture, initialization strategy, or hyperparameter tuning, and that medical-domain fine-tuning of the autoencoder does not close it. To probe and partially narrow the gap, we introduce noise-conditioned latent classifiers with FiLM layers and image-space distillation, deploying them as diagnostic instruments and initial mitigations to evaluate latent space quality. These models offer {\$}{\$}64{\{}{\backslash}times {\}}{\$}{\$}64{\texttimes}throughput and {\$}{\$}120{\{}{\backslash}times {\}}{\$}{\$}120{\texttimes}memory gains over image-space models while serving as diagnostic tools for latent space quality. Our analysis provides a new framework for evaluating autoencoder latent spaces and identifies their structure, rather than their fidelity or domain specificity, as the primary obstacle to closing the performance gap between real and synthetic medical training data. Model and code made available at https://github.com/MischaD/LearnabilityGap.git.},
	address = {Cham},
	author = {Dombrowski, Mischa and N{\"u}tzel, Felix and Kainz, Bernhard},
	booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
	editor = {Yang, Guang and Adeli, Ehsan and de Bruijne, Marleen and Papie{\.{z}}, Bart{\l}omiej W. and Speidel, Stefanie and Tiwari, Pallavi and Zheng, Guoyan and Yaqub, Mohammad and Dou, Qi and Rekik, Islem},
	isbn = {978-3-032-38189-7},
	pages = {544--554},
	publisher = {Springer Nature Switzerland},
	title = {The Learnability Gap in Medical Latent Diffusion},
	year = {2027}}
```
