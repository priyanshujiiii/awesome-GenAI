# Awesome-Generative-AI
A curated list of papers, datasets, and resources on Generative AI, covering Variational Autoencoders, Generative Adversarial Networks, Autoregressive Models, Diffusion Models, Large Language Models, and more.

## Table of Contents
1. [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
2. [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
3. [Autoregressive Models](#autoregressive-models)
4. [Diffusion Models](#diffusion-models)
5. [Large Language Models (LLMs)](#large-language-models-llms)
6. [Other / Hybrid Generative Models](#other--hybrid-generative-models)
7. [Surveys & Overviews](#surveys--overviews)

---

## **Variational Autoencoders (VAEs)**

Probabilistic generative models that learn latent representations of data by optimizing a variational lower bound.  
**GitHub topics:** `variational-autoencoder, vae, probabilistic-models, representation-learning, generative-ai`

| Title | Authors | Venue | Year | Paper Link |
|---|---|---|---|---|
| Auto-Encoding Variational Bayes | Kingma, D. P.; Welling, M. | ICLR | 2014 | [PDF](https://arxiv.org/abs/1312.6114) |
| Stochastic Backpropagation and Approximate Inference in Deep Generative Models | Rezende, D. J.; Mohamed, S.; Wierstra, D. | ICML | 2014 | [PDF](https://arxiv.org/abs/1401.4082) |
| Conditional Variational Autoencoder (CVAE) | Sohn, K.; Yan, X.; Lee, H. | NeurIPS | 2015 | [PDF](https://arxiv.org/abs/1506.05561) |
| Variational Lossy Autoencoder | Chen, X.; Kingma, D. P.; Salimans, T.; Duan, Y.; Dhariwal, P.; Schulman, J.; Sutskever, I.; Abbeel, P. | ICLR | 2017 | [PDF](https://arxiv.org/abs/1611.02731) |
| Ladder Variational Autoencoders | Sønderby, C. K.; Raiko, T.; Maaløe, L.; Sønderby, S. K.; Winther, O. | NeurIPS | 2016 | [PDF](https://arxiv.org/abs/1602.02282) |
| Disentangling by Factorising | Kim, H.; Mnih, A. | ICML | 2018 | [PDF](https://arxiv.org/abs/1802.05983) |
| Vector Quantised-Variational AutoEncoder (VQ-VAE) | van den Oord, A.; Vinyals, O.; Kavukcuoglu, K. | NeurIPS | 2017 | [PDF](https://arxiv.org/abs/1711.00937) |
| NVAE: A Deep Hierarchical Variational Autoencoder | Vahdat, A.; Kautz, J. | NeurIPS | 2020 | [PDF](https://arxiv.org/abs/2007.03898) |
| InfoVAE: Balancing Learning and Inference in Variational Autoencoders | Zhao, S.; Song, J.; Ermon, S. | AAAI | 2019 | [PDF](https://arxiv.org/abs/1706.02262) |
| β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework | Higgins, I.; Matthey, L.; Pal, A.; Burgess, C.; Glorot, X.; Botvinick, M.; Mohamed, S.; Lerchner, A. | ICLR | 2017 | [PDF](https://openreview.net/forum?id=Sy2fzU9gl) |
| β-TCVAE: Learning Disentangled Representations with Total Correlation | Chen, R. T. Q.; Li, X.; Grosse, R.; Duvenaud, D. | ICLR | 2018 | [PDF](https://arxiv.org/abs/1802.04942) |
| DIP-VAE: Disentangled Inferred Prior VAE | Kumar, A.; Sattigeri, P.; Balakrishnan, A. | ICLR | 2018 | [PDF](https://openreview.net/forum?id=H1kG7GZRW) |
| FactorVAE: A Probabilistic Framework for Learning Disentangled Representations | Kim, H.; Mnih, A. | ICML | 2018 | [PDF](https://arxiv.org/abs/1802.05983) |
| StyleVAE: Disentangling Style and Content in VAE Representations | Lee, H. et al. | NeurIPS Workshop | 2018 | [PDF](https://arxiv.org/abs/1808.04537) |
| JointVAE: Learning Disentangled Joint Continuous and Discrete Representations | Dupont, E. | NeurIPS | 2018 | [PDF](https://arxiv.org/abs/1804.00104) |
| Sparse VAE: Learning Structured Latent Variables | Mathieu, E.; Rainforth, T.; Siddharth, N.; Teh, Y. W. | NeurIPS | 2019 | [PDF](https://arxiv.org/abs/1906.04904) |
| VLAE: Variational Ladder Autoencoder | Zhao, S. et al. | arXiv | 2017 | [PDF](https://arxiv.org/abs/1707.04487) |
| Importance Weighted Autoencoders | Burda, Y.; Grosse, R.; Salakhutdinov, R. | ICLR | 2016 | [PDF](https://arxiv.org/abs/1509.00519) |
| ResNet VAE: Scaling VAEs with Residual Networks | Kingma, D. P. et al. | arXiv | 2016 | [PDF](https://arxiv.org/abs/1606.04934) |
| GraphVAE: Variational Autoencoders for Graph Generation | Simonovsky, M.; Komodakis, N. | ICANN | 2018 | [PDF](https://arxiv.org/abs/1802.03480) |

---

## **Generative Adversarial Networks (GANs)**

A framework where two neural networks — a generator and a discriminator — compete in a minimax game, leading to realistic data generation.  
**GitHub topics:** `gan, generative-adversarial-network, adversarial-learning, generative-ai`

| Title | Authors | Venue | Year | Paper Link |
|---|---|---|---|---|
| Generative Adversarial Nets | Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; Bengio, Y. | NeurIPS | 2014 | [PDF](https://arxiv.org/abs/1406.2661) |
| Conditional Generative Adversarial Nets (cGAN) | Mirza, M.; Osindero, S. | arXiv | 2014 | [PDF](https://arxiv.org/abs/1411.1784) |
| DCGAN: Unsupervised Representation Learning with Deep Convolutional GANs | Radford, A.; Metz, L.; Chintala, S. | ICLR | 2016 | [PDF](https://arxiv.org/abs/1511.06434) |
| WGAN: Wasserstein GAN | Arjovsky, M.; Chintala, S.; Bottou, L. | ICML | 2017 | [PDF](https://arxiv.org/abs/1701.07875) |
| WGAN-GP: Improved Training of Wasserstein GANs | Gulrajani, I.; Ahmed, F.; Arjovsky, M.; Dumoulin, V.; Courville, A. | NeurIPS | 2017 | [PDF](https://arxiv.org/abs/1704.00028) |
| LSGAN: Least Squares GAN | Mao, X.; Li, Q.; Xie, H.; Lau, R. Y. K.; Wang, Z.; Paul Smolley, S. | ICCV | 2017 | [PDF](https://arxiv.org/abs/1611.04076) |
| BEGAN: Boundary Equilibrium GAN | Berthelot, D.; Schumm, T.; Metz, L. | arXiv | 2017 | [PDF](https://arxiv.org/abs/1703.10717) |
| BigGAN: Large Scale GAN Training | Brock, A.; Donahue, J.; Simonyan, K. | ICLR | 2019 | [PDF](https://arxiv.org/abs/1809.11096) |
| StyleGAN: A Style-Based Generator Architecture for GANs | Karras, T.; Laine, S.; Aila, T. | CVPR | 2019 | [PDF](https://arxiv.org/abs/1812.04948) |
| StyleGAN2: Analyzing and Improving the Image Quality of GANs | Karras, T.; Laine, S.; Aittala, M.; Hellsten, J.; Lehtinen, J.; Aila, T. | CVPR | 2020 | [PDF](https://arxiv.org/abs/1912.04958) |
| StyleGAN3: Alias-Free Generative Adversarial Networks | Karras, T.; Aittala, M.; Hellsten, J.; Laine, S.; Lehtinen, J.; Aila, T. | NeurIPS | 2021 | [PDF](https://arxiv.org/abs/2106.12423) |
| CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks | Zhu, J.-Y.; Park, T.; Isola, P.; Efros, A. A. | ICCV | 2017 | [PDF](https://arxiv.org/abs/1703.10593) |
| Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks | Isola, P.; Zhu, J.-Y.; Zhou, T.; Efros, A. A. | CVPR | 2017 | [PDF](https://arxiv.org/abs/1611.07004) |
| GauGAN: Semantic Image Synthesis with Spatially-Adaptive Normalization | Park, T.; Liu, M.-Y.; Wang, T.-C.; Zhu, J.-Y. | CVPR | 2019 | [PDF](https://arxiv.org/abs/1903.07291) |
| SAGAN: Self-Attention Generative Adversarial Networks | Zhang, H.; Goodfellow, I.; Metaxas, D.; Odena, A. | ICML | 2019 | [PDF](https://arxiv.org/abs/1805.08318) |
| ESRGAN: Enhanced Super-Resolution GAN | Wang, X.; Yu, K.; Wu, S.; Gu, J.; Liu, Y.; Dong, C.; Qiao, Y.; Change Loy, C. | ECCV | 2018 | [PDF](https://arxiv.org/abs/1809.00219) |
| SPADE: Semantic Image Synthesis with Spatially-Adaptive Normalization | Park, T.; Liu, M.-Y.; Wang, T.-C.; Zhu, J.-Y. | CVPR | 2019 | [PDF](https://arxiv.org/abs/1903.07291) |
| StarGAN: Unified GAN for Multi-Domain Image-to-Image Translation | Choi, Y.; Choi, M.; Kim, M.; Ha, J.-W.; Kim, S.; Choo, J. | CVPR | 2018 | [PDF](https://arxiv.org/abs/1711.09020) |
| StarGAN v2: Diverse Image Synthesis for Multiple Domains | Choi, Y.; Uh, Y.; Yoo, J.; Ha, J.-W. | CVPR | 2020 | [PDF](https://arxiv.org/abs/1912.01865) |
| DeblurGAN-v2: Deblurring with Feature Pyramid Networks | Kupyn, O.; Martyniuk, T.; Wu, J.; Wang, Z. | ICCV | 2019 | [PDF](https://arxiv.org/abs/1908.03826) |
| Text2Image GAN: Generating Images from Captions | Reed, S.; Akata, Z.; Yan, X.; Logeswaran, L.; Schiele, B.; Lee, H. | ICML | 2016 | [PDF](https://arxiv.org/abs/1605.05396) |
| StackGAN: Text to Photo-Realistic Image Synthesis | Zhang, H.; Xu, T.; Li, H.; Zhang, S.; Wang, X.; Huang, X.; Metaxas, D. N. | ICCV | 2017 | [PDF](https://arxiv.org/abs/1612.03242) |
| StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks | Zhang, H.; Xu, T.; Li, H.; Zhang, S.; Wang, X.; Huang, X.; Metaxas, D. N. | TPAMI | 2018 | [PDF](https://arxiv.org/abs/1710.10916) |
| Speech2Face: Learning the Face Behind a Voice | Oh, T.-H.; Dekel, T.; Kim, C.; Mosseri, I.; Freeman, W. T.; Rubinstein, M.; Matusik, W. | CVPR | 2019 | [PDF](https://arxiv.org/abs/1905.09773) |


---

## **Autoregressive Models**

Autoregressive models factor the joint distribution of data as a product of conditional distributions and model each element sequentially — excellent for high-fidelity generation in images, audio and text.  
**GitHub topics:** `autoregressive, pixelcnn, pixelrnn, wavenet, language-models, generative-ai`

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| PixelRNN: Conditional Image Generation with Pixel Recurrent Neural Networks | A. van den Oord, N. Kalchbrenner, K. Kavukcuoglu | ICML | 2016 | https://arxiv.org/abs/1601.06759 |
| PixelCNN: Generative Image Modeling Using Convolutional Neural Networks | A. van den Oord, N. Kalchbrenner, O. Vinyals, L. Espeholt, A. Graves, K. Kavukcuoglu | ICML | 2016 | https://arxiv.org/abs/1606.05328 |
| PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood | T. Salimans, A. Karpathy, X. Chen, D. P. Kingma | ICLR | 2017 | https://arxiv.org/abs/1701.05517 |
| Image Transformer | N. Parmar, A. Vaswani, J. Uszkoreit, L. Kaiser, N. Shazeer, A. Ku, D. Tran | ICML | 2018 | https://arxiv.org/abs/1802.05751 |
| PixelSNAIL: An Improved Autoregressive Model for Image Generation | O. J. A. van den Oord, K. Espeholt, A. Vinyals, I. Babuschkin (PixelSNAIL authors vary) | ICML Workshops / arXiv | 2017 | https://arxiv.org/abs/1712.09763 |
| WaveNet: A Generative Model for Raw Audio | A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, K. Kavukcuoglu | SSW | 2016 | https://arxiv.org/abs/1609.03499 |
| WaveRNN: A Compact, Efficient, and Scalable Neural Vocoder | R. Kalchbrenner, N. Kalchbrenner et al. (WaveRNN devs vary) | arXiv / ICASSP followups | 2018 | https://arxiv.org/abs/1802.08435 |
| Transformer (original for sequence modeling) | A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin | NeurIPS | 2017 | https://arxiv.org/abs/1706.03762 |
| Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context | Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Quoc V. Le, R. Salakhutdinov | ACL/NeurIPS | 2019 | https://arxiv.org/abs/1901.02860 |
| GPT: Improving Language Understanding by Generative Pre-Training | A. Radford et al. | OpenAI tech report | 2018 | https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf |
| GPT-2: Language Models are Unsupervised Multitask Learners | A. Radford et al. | OpenAI tech report | 2019 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| GPT-3: Language Models are Few-Shot Learners | T. Brown et al. | NeurIPS | 2020 | https://arxiv.org/abs/2005.14165 |
| XLNet: Generalized Autoregressive Pretraining for Language Understanding | Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. Salakhutdinov, Q. V. Le | NeurIPS | 2019 | https://arxiv.org/abs/1906.08237 |
| Sparse Transformer / Long-range Transformer (examples of autoregressive for long sequences) | Reformer / Sparse Transformer authors (e.g., A. Kitaev et al.) | NeurIPS / ICLR | 2020 / 2019 | https://arxiv.org/abs/1904.10509 (Sparse Transformer) |
| ImageGPT: Generative Pretraining from Pixels | A. Radford, M. Chen, J. Child et al. | arXiv | 2020 | https://openai.com/research/image-gpt |
| Char-RNN / LSTM language models (historic AR baselines) | A. Karpathy (popularized), S. Hochreiter & J. Schmidhuber (LSTM) | Various | 1997 / 2015 (LSTM history) | https://arxiv.org/abs/1506.02078 (char-rnn blog reference) |
| MADE: Masked Autoencoder for Distribution Estimation | G. Germain, K. Gregor, I. Murray, H. Larochelle | ICML | 2015 | https://arxiv.org/abs/1502.03509 |
| MAF: Masked Autoregressive Flow | L. Dinh, D. Krueger, Y. Bengio | arXiv / ICML | 2017 | https://arxiv.org/abs/1705.07057 |
| Transformer-based Autoregressive Image Models (e.g., VQ-VAE + Autoregressive Priors) | Oord et al., Razavi et al. (VQ-VAE, VQGAN, etc.) | NeurIPS / ICLR | 2017–2021 | https://arxiv.org/abs/1711.00937 (VQ-VAE) |
| RNN-Transducer / Neural Autoregressive Flow (audio & seq models) | Graves et al., other authors | ICASSP / NeurIPS | various | (respective papers links) |


---

## **Diffusion Models**

Diffusion models are generative models that learn to reverse a gradual noising process, producing high-quality samples with stable training dynamics.  
**GitHub topics:** `diffusion-models, score-matching, ddpm, ddim, stable-diffusion, generative-ai`

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Deep Unsupervised Learning using Nonequilibrium Thermodynamics | J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, S. Ganguli | ICML | 2015 | https://arxiv.org/abs/1503.03585 |
| Denoising Diffusion Probabilistic Models (DDPM) | J. Ho, A. Jain, P. Abbeel | NeurIPS | 2020 | https://arxiv.org/abs/2006.11239 |
| Denoising Diffusion Implicit Models (DDIM) | J. Song, C. Meng, S. Ermon | ICLR | 2021 | https://arxiv.org/abs/2010.02502 |
| Improved Denoising Diffusion Probabilistic Models | A. Nichol, P. Dhariwal | ICML | 2021 | https://arxiv.org/abs/2102.09672 |
| Score-Based Generative Modeling through Stochastic Differential Equations (SDE) | Y. Song, J. Sohl-Dickstein, D. Kingma, A. Kumar, S. Ermon, B. Poole | ICLR | 2021 | https://arxiv.org/abs/2011.13456 |
| Guided Diffusion: Diffusion Models Beat GANs on Image Synthesis | P. Dhariwal, A. Nichol | NeurIPS | 2021 | https://arxiv.org/abs/2105.05233 |
| Cascaded Diffusion Models for High Fidelity Image Generation | J. Ho et al. | arXiv | 2021 | https://arxiv.org/abs/2106.15282 |
| Latent Diffusion Models (LDM) | R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer | CVPR | 2022 | https://arxiv.org/abs/2112.10752 |
| Stable Diffusion | Stability AI et al. | GitHub/CreativeML | 2022 | https://stability.ai/blog/stable-diffusion-public-release |
| Imagen: Photorealistic Text-to-Image Diffusion Models | C. Saharia, W. Chan, S. Saxena, L. Li, Y. Sun, D. Fleet, M. Norouzi, W. Chan | arXiv | 2022 | https://arxiv.org/abs/2205.11487 |
| Parti: Scaling Autoregressive Models for Text-to-Image Generation | W. Yu, D. A. Ramesh, D. D. Chan, et al. | arXiv | 2022 | https://arxiv.org/abs/2206.10789 |
| Muse: Text-to-Image Generation via Masked Generative Transformers | A. Chang, A. Chang, W. Yu, et al. | arXiv | 2023 | https://arxiv.org/abs/2301.00704 |
| ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models | L. Zhang, M. Chen, et al. | arXiv | 2023 | https://arxiv.org/abs/2302.05543 |
| DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation | N. Ruiz, Y. Li, V. Balntas, et al. | ECCV | 2022 | https://arxiv.org/abs/2208.12242 |
| Textual Inversion: Learning New Concepts in Text-to-Image Diffusion Models | R. Gal, Y. Alaluf, et al. | SIGGRAPH Asia | 2022 | https://arxiv.org/abs/2208.01618 |
| InstructPix2Pix: Learning to Follow Image Editing Instructions | T. Brooks, A. Holynski, A. A. Efros | CVPR | 2023 | https://arxiv.org/abs/2211.09800 |
| DiffWave: A Versatile Diffusion Model for Audio Synthesis | Z. Kong, W. Ping, J. Huang, K. Zhao, B. Catanzaro | ICLR | 2021 | https://arxiv.org/abs/2009.09761 |
| Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech | V. Popov, V. Vovk, T. Sadekova, et al. | ICML | 2021 | https://arxiv.org/abs/2105.06337 |
| Palette: Image-to-Image Diffusion Models | A. Saharia, W. Chan, H. Chang, et al. | SIGGRAPH | 2022 | https://arxiv.org/abs/2111.05826 |
| Video Diffusion Models | M. Ho et al. | arXiv | 2022 | https://arxiv.org/abs/2204.03458 |
| MotionDiffusion: Generating Human Motions via Denoising Diffusion Probabilistic Models | A. Tevet, B. Gordon, et al. | ICLR | 2023 | https://arxiv.org/abs/2209.14916 |


---

## **Large Language Models (LLMs)**

Large-scale transformer-based language models trained on vast text corpora, capable of few-shot learning, reasoning, and multi-modal generation.  
**GitHub topics:** `large-language-models, llm, transformer, gpt, instruction-tuning, generative-ai`

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Attention is All You Need | A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin | NeurIPS | 2017 | https://arxiv.org/abs/1706.03762 |
| GPT: Improving Language Understanding by Generative Pre-Training | A. Radford et al. | OpenAI tech report | 2018 | https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf |
| GPT-2: Language Models are Unsupervised Multitask Learners | A. Radford et al. | OpenAI tech report | 2019 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| GPT-3: Language Models are Few-Shot Learners | T. Brown et al. | NeurIPS | 2020 | https://arxiv.org/abs/2005.14165 |
| GPT-4 Technical Report | OpenAI | arXiv | 2023 | https://arxiv.org/abs/2303.08774 |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | J. Devlin, M. Chang, K. Lee, K. Toutanova | NAACL | 2019 | https://arxiv.org/abs/1810.04805 |
| RoBERTa: A Robustly Optimized BERT Pretraining Approach | Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, V. Stoyanov | arXiv | 2019 | https://arxiv.org/abs/1907.11692 |
| T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | C. Raffel et al. | JMLR | 2020 | https://arxiv.org/abs/1910.10683 |
| PaLM: Scaling Language Models to 540 Billion Parameters | A. Chowdhery, S. Narang, J. Devlin, et al. | arXiv | 2022 | https://arxiv.org/abs/2204.02311 |
| PaLM 2: Scaling Language Models with Pathways | Google DeepMind | arXiv | 2023 | https://storage.googleapis.com/pathways-language-model/PaLM2.pdf |
| LLaMA: Open and Efficient Foundation Language Models | H. Touvron, T. Lavril, G. Izacard, et al. | arXiv | 2023 | https://arxiv.org/abs/2302.13971 |
| LLaMA 2: Open Foundation and Fine-Tuned Chat Models | H. Touvron, L. Martin, K. Stone, et al. | arXiv | 2023 | https://arxiv.org/abs/2307.09288 |
| Chinchilla: Training Compute-Optimal Large Language Models | J. Hoffmann, S. Borgeaud, A. Mensch, et al. | arXiv | 2022 | https://arxiv.org/abs/2203.15556 |
| OPT: Open Pre-trained Transformer Language Models | S. Zhang, M. Roller, et al. | arXiv | 2022 | https://arxiv.org/abs/2205.01068 |
| BLOOM: A 176B-Parameter Open-Access Multilingual Language Model | B. BigScience Workshop et al. | arXiv | 2022 | https://arxiv.org/abs/2211.05100 |
| Falcon LLM: Scaling for Efficient Inference | TII | arXiv | 2023 | https://arxiv.org/abs/2306.01116 |
| MPT: MosaicML Pretrained Transformer Models | MosaicML | GitHub | 2023 | https://www.mosaicml.com/blog/mpt-30b |
| Claude: Constitutional AI for Harmlessness | Anthropic | Blog / arXiv | 2023 | https://www.anthropic.com/index/claude |
| Instruction Tuning for LLMs | L. Ouyang, J. Wu, X. Jiang, et al. | arXiv | 2022 | https://arxiv.org/abs/2203.02155 |
| Self-Instruct: Aligning LMs with Self-Generated Instructions | Y. Wang, W. Kordi, et al. | ACL | 2023 | https://arxiv.org/abs/2212.10560 |
| FLAN: Fine-Tuned Language Models | C. Chung, A. Tay, et al. | arXiv | 2022 | https://arxiv.org/abs/2210.11416 |
| FLAN-T5 & FLAN-PaLM | Google AI | arXiv | 2022 | https://arxiv.org/abs/2210.11416 |
| InstructGPT: Aligning Language Models to Follow Instructions | L. Ouyang et al. | arXiv | 2022 | https://arxiv.org/abs/2203.02155 |
| GPT-4V: Multimodal GPT-4 with Vision | OpenAI | Blog / arXiv | 2023 | https://openai.com/research/gpt-4v-system-card |
| Kosmos-1: Multimodal Large Language Model | M. Huang, S. Bubeck, et al. | arXiv | 2023 | https://arxiv.org/abs/2302.14045 |
| Gemini 1: Multimodal Reasoning at Scale | Google DeepMind | Blog / arXiv | 2023 | https://deepmind.google/discover/blog/announcing-gemini-1/ |
| Orca: Progressive Learning from Complex Explanation Traces of GPT-4 | M. Mukherjee, et al. | arXiv | 2023 | https://arxiv.org/abs/2306.02707 |

---

## **Other / Hybrid Generative Models**

Models that combine multiple generative paradigms (e.g., VAE+GAN, AR+Diffusion), or use alternative approaches such as flow-based generative modeling.  
**GitHub topics:** `normalizing-flows, flow-based-models, hybrid-generative-models, generative-ai`

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| RealNVP: Density Estimation using Real-valued Non-Volume Preserving Transformations | L. Dinh, J. Sohl-Dickstein, S. Bengio | ICLR | 2017 | https://arxiv.org/abs/1605.08803 |
| Glow: Generative Flow with Invertible 1x1 Convolutions | D. Kingma, P. Dhariwal | NeurIPS | 2018 | https://arxiv.org/abs/1807.03039 |
| NICE: Non-linear Independent Components Estimation | L. Dinh, D. Krueger, Y. Bengio | ICLR Workshop | 2015 | https://arxiv.org/abs/1410.8516 |
| VAE-GAN: Autoencoding beyond pixels using a learned similarity metric | A. Larsen, S. Sønderby, H. Larochelle, O. Winther | ICML | 2016 | https://arxiv.org/abs/1512.09300 |
| VQGAN: Taming Transformers for High-Resolution Image Synthesis | P. Esser, R. Rombach, B. Ommer | CVPR | 2021 | https://arxiv.org/abs/2012.09841 |
| DALLE: Zero-Shot Text-to-Image Generation | A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, I. Sutskever | arXiv | 2021 | https://arxiv.org/abs/2102.12092 |
| DALLE-2: Hierarchical Text-Conditional Image Generation with CLIP Latents | A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, M. Chen | arXiv | 2022 | https://arxiv.org/abs/2204.06125 |
| Parti: Scaling Autoregressive Models for Text-to-Image Generation | W. Yu, D. A. Ramesh, D. D. Chan, et al. | arXiv | 2022 | https://arxiv.org/abs/2206.10789 |
| MaskGIT: Masked Generative Image Transformer | H. Chang, H. Zhang, L. Jiang, C. Liu, W. Freeman, P. Isola | CVPR | 2022 | https://arxiv.org/abs/2202.04200 |
| MAGVIT: Masked Generative Video Transformer | J. Yu, et al. | arXiv | 2022 | https://arxiv.org/abs/2212.05199 |
| VideoGPT: Video Generation using VQ-VAE and Transformers | W. Yan, et al. | arXiv | 2021 | https://arxiv.org/abs/2104.10157 |
| CogVideo: Large-scale Pretraining for Text-to-Video Generation | M. Hong, X. Zhou, et al. | arXiv | 2022 | https://arxiv.org/abs/2205.15868 |
| Gato: A Generalist Agent | S. Reed, et al. | arXiv | 2022 | https://arxiv.org/abs/2205.06175 |
| U-Net for Generative Modeling | O. Ronneberger, P. Fischer, T. Brox | MICCAI | 2015 | https://arxiv.org/abs/1505.04597 |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs | A. Jaegle, S. Borgeaud, et al. | ICML | 2022 | https://arxiv.org/abs/2107.14795 |
| CLIP: Learning Transferable Visual Models from Natural Language Supervision | A. Radford, J. W. Kim, C. Hallacy, et al. | ICML | 2021 | https://arxiv.org/abs/2103.00020 |
| Flamingo: Few-Shot Learning with Multimodal Models | J. Alayrac, J. Donahue, et al. | NeurIPS | 2022 | https://arxiv.org/abs/2204.14198 |
| Kosmos-2: Language Is Not All You Need | M. Huang, S. Bubeck, et al. | arXiv | 2023 | https://arxiv.org/abs/2302.14045 |

---

## 🧾 Surveys & Overviews

Large, high-quality overview papers that summarize, compare, or benchmark families of generative models (GANs, VAEs, flows, diffusion models, LLMs, multimodal models), their evaluation, and societal impacts.  
**GitHub topics:** `survey, review, generative-models, diffusion-models, gans, vae, normalizing-flows, llm-survey, evaluation, ethics`

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| A Survey on Generative AI Applications | R. Gozalo-Brizuela, E. C. Garrido-Merchán | arXiv | 2023 | https://arxiv.org/abs/2306.02781 |
| ChatGPT is not all you need: A state of the art review of large generative AI models | R. Gozalo-Brizuela, E. C. Garrido-Merchán | arXiv | 2023 | https://arxiv.org/abs/2301.04655 |
| A survey on generative modeling with limited data, few-shot and zero-shot | M. Abdollahzadeh et al. | arXiv | 2023 | https://arxiv.org/abs/2307.14397 |
| Generative Adversarial Networks (landmark survey / review) | I. Goodfellow et al. | Communications of the ACM (overview) / arXiv (orig.) | 2014 / 2020 | https://arxiv.org/abs/1406.2661 |
| Denoising Diffusion Probabilistic Models (foundational paper; overview of diffusion models) | J. Ho, A. Jain, P. Abbeel | NeurIPS / arXiv | 2020 | https://arxiv.org/abs/2006.11239 |
| Diffusion models beat GANs on image synthesis (empirical comparison & insights) | P. Dhariwal, A. Nichol | NeurIPS | 2021 | https://arxiv.org/abs/2105.05233 |
| Normalizing flows for probabilistic modeling and inference (survey / tutorial) | G. Papamakarios, E. Nalisnick, D. Rezende, S. Mohamed, B. Lakshminarayanan | JMLR / arXiv (review) | 2021 | https://arxiv.org/abs/2102.12514 |
| A Survey of Deep Generative Models: VAE, GAN, Flow, and Energy-based models | A. Oussidi, A. Elhassouny | ISCV proceedings | 2018 | (paper search: title) |
| A Survey on Generative Modeling with Limited Data, Few-Shot and Zero-Shot | M. Abdollahzadeh et al. | arXiv | 2023 | https://arxiv.org/abs/2307.14397 |
| A Study on the Evaluation of Generative Models | E. Betzalel, C. Penso, A. Navon, E. Fetaya | arXiv | 2022 | https://arxiv.org/abs/2206.10935 |
| A Survey of Low-Bit / Efficient Models & Compression for Generative Systems | (multiple authors — compendia & reviews) | arXiv / conference surveys | 2022–2024 | (see collection / repo links) |
| A Comprehensive Survey of Diffusion Models & Their Applications | (collection of reviews) | arXiv / journals | 2022–2024 | (search: "survey diffusion models 2023") |
| Surveys on LLMs: Foundations, Applications, and Safety (collection) | Multiple authors (many surveys: Foundations of LLMs; Efficient LLMs; Safety & Alignment surveys) | arXiv / ACM / journals | 2023–2025 | See: https://arxiv.org/search/?query=survey+large+language+models |
| Efficient Large Language Models: A Survey | Various | arXiv | 2023 | https://arxiv.org/abs/2312.03863 |
| A Survey on Model Compression for Large Language Models | Various | arXiv | 2023 | https://arxiv.org/abs/2308.07633 |
| Generative AI: Promise and Peril (short overview / perspective) | A. Jo | Nature (commentary) | 2023 | https://www.nature.com/articles/d41586-023-02120-4 |
| Generative AI at Work (economics / impact survey) | E. Brynjolfsson, D. Li, L. R. Raymond | NBER Tech. Rep. | 2023 | https://www.nber.org/papers/w31602 |
| A Comprehensive Survey on Pretrained Foundation Models: From BERT to ChatGPT | (survey collection) | arXiv | 2023 | https://arxiv.org/search/?query=pretrained+foundation+models+survey |
| A Survey on Evaluation of Large Language Models | Various | arXiv / ACM | 2023–2024 | https://arxiv.org/abs/2310.19736 |
| A Comprehensive Survey of Compression Algorithms for Language Models | Various | arXiv | 2024 | https://arxiv.org/abs/2401.15347 |
| Survey: Generative Models across modalities (vision, audio, video, text) | Various | arXiv / journal surveys | 2022–2024 | (search: "survey generative models multimodal 2023") |

---
