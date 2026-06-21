# The Learnability Gap in Medical Latent Diffusion

**Accepted at MICCAI 2026** — See you in Strasbourg! 🇫🇷

Generative data augmentation with latent diffusion models is a promising strategy for addressing class imbalance in medical imaging, yet current approaches focus on perceptual fidelity and domain-specific autoencoder fine-tuning while neglecting a more fundamental bottleneck. 

We identify and formalize the *learnability gap*: large-scale pretrained autoencoders faithfully encode discriminative features for medical classification, as evidenced by near-lossless performance in reconstruction space, yet their latent representations are structured in ways that are difficult for classifiers to learn from. 

Across five autoencoder families and four medical benchmarks spanning chest radiography, dermatoscopy, computed tomography, and echocardiography, we show that this gap persists regardless of architecture, initialization strategy, or hyperparameter tuning, and that medical-domain fine-tuning of the autoencoder does not close it. 

To probe and partially narrow the gap, we develop noise-conditioned latent classifiers with FiLM layers and image-space distillation that offer **64× throughput** and **120× memory gains** over image-space models while serving as diagnostic tools for latent space quality. Our analysis provides a new framework for evaluating autoencoder latent spaces and identifies their structure, rather than their fidelity or domain specificity, as the primary obstacle to closing the performance gap between real and synthetic medical training data.

### 📄 Preprint
*Preprint available!* 

```
@misc{dombrowski2026learnabilitygapmedicallatent,
      title={The Learnability Gap in Medical Latent Diffusion}, 
      author={Mischa Dombrowski and Felix Nützel and Bernhard Kainz},
      year={2026},
      eprint={2605.17087},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.17087}, 
}
```
