# EMGAN - Da fala ao EMG

# EMGAN - From speech to EMG

## Presentation

This project originated in the context of the graduate course _IA376N - Generative AI: from models to multimodal applications_,
offered in the first semester of 2026, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

|Name | RA | Specialization|
|--|--|--|
| Daniel Neto | 169408 | Computer Engineering|
| Enzo Campos | 247069 | Computer Engineering|
| Marcelo Ferreira | 300882 | Computer Engineering|

## Project Summary Description

Surface Electromyography (sEMG) signals of articulatory muscles reflect the speech production process. As such, they are a biosignal of interest for Silent Speech Interfaces (SSIs) [1], which aim to enable speech communication without depending on acoustic speech. However, acquiring these biological signals is notoriously difficult. Subjects must undergo a tiresome recording procedure, resulting in small datasets with low subject variability. Furthermore, the collected data is often lawfully restricted. Consequently, there is a strong motivation for Speech-to-EMG (STE) modeling, which could be explored to generate new, artificial EMG signals to improve ETS model training and mitigate data scarcity.

To address the issues of data acquisition and aid in the development of robust SSIs, we propose a method of generating synthetic EMG signals based on the STE-GAN architecture [2]. The main goal of this project is to generate reliable EMG data from acoustic speech that is not only similar to the target domain, but also capable of retaining high linguistic accuracy (e.g., maintaining a low Word Error Rate) after being converted back to audio.

The generator outputs a C-channel EMG signal. Therefore, the output of the generative model will be a multi-channel EMG signal of the exact same dimension as the target input data.

[PRESENTATION LINK](https://docs.google.com/presentation/d/1RQUPtrMktuhXLVYFDq8A5LLMe22JyeH8VmcdLbCdixM/edit?usp=sharing)

## Proposed Methodology

For this project, our models require datasets containing paired EMG and audio signals. Since this type of data is relatively rare, we have surveyed the publicly available datasets to compose a preliminary list of candidates. Over the first three weeks of the project, we will conduct an in-depth analysis of these sources to evaluate their quality, compatibility, and suitability for our goals. Following this assessment, we will select the final subset of datasets to be utilized and define how they will be integrated into our workflow. The datasets currently under consideration are:

- The EMG-UKA corpus for electromyographic speech processing [3]
- Digital Voicing of Silent Speech [4]
- An open dataset of multidimensional signals based on different speech patterns in pragmatic Mandarin [5]
- DiffMV-ETS: Diffusion-based Multi-Voice Electromyography-to-Speech Conversion using Speaker-Independent Speech Training Targets [6]
- AVE Speech Dataset: A Comprehensive Benchmark for Multi-Modal Speech Recognition Integrating Audio, Visual, and Electromyographic Signals [7]
- CSL-EMG_Array: An Open Access Corpus for EMG-to-Speech Conversion [8]
- emg2speech: synthesizing speech from electromyography using self-supervised speech models [9]
- SilentWear: an Ultra-Low Power Wearable System for EMG-based Silent Speech Recognition [10]
- Sentence-Level Silent Speech Recognition Using a Wearable EMG/EEG Sensor System with AI-Driven Sensor Fusion and Language Model [11]

The generative modeling approach that will serve as the baseline for this study is the Speech-to-Electromyography Generative Adversarial Network (STE-GAN) presented in [2]. We found this approach particularly compelling due to its strong and reproducible results. Notably, STE-GAN directly converts acoustic speech to EMG signals in an end-to-end fashion, eliminating the need to predict intermediate features. In doing so, it achieved impressive metrics, such as a high Envelope Correlation Coefficient of 0.66 and over 80% Phoneme Accuracy on the generated signals. Furthermore, instead of naively sampling from a Gaussian distribution, the model conditions the generation on a controllable latent space—soft speech units extracted from audio—which importantly enables the model to generalize to speech of unseen speakers. Finally, our choice is strongly motivated by the fact that the authors made their code openly available, which greatly facilitates reproducibility for our project.

All articles used as references are cited in the bibliography. Other bibliography might be added throughout the development of the project.

For the project's development, our main codebase will be built in Python, leveraging libraries such as PyTorch, NumPy, and other data science and signal processing tools. We plan to use Jupyter Notebooks for interactive coding and evaluation, while GitHub will handle version control and code hosting. Finally, to ensure the models are trained within a feasible timeframe, we will rely on GPU computing, specifically an NVIDIA RTX 5070 Ti (16 GiB), with the possibility of incorporating additional GPUs as needed.

At first, we aim to reproduce the results presented in [2]. Then, by evaluating the model on additional datasets and testing architectural modification hypotheses, we expect to both consolidate the architecture for this type of data and improve its evaluation scores.

Guided by the metrics presented in [2], we plan to measure the correlation between synthetic and real EMG envelope data, as well as the correctness of generated speech from synthetic EMG using the WER metric (usability). Additional metrics may also be proposed to evaluate any architectural changes.

## Schedule

**WEEKS 1 to 3**: During this initial phase, we will conduct literature reviews on the core concepts underlying the problem and the baseline model. Simultaneously, we will evaluate the 9 preliminary datasets. By assigning one dataset per member each week, we will thoroughly review all candidates within this timeframe. To build specialized expertise, individual research tasks will be divided as follows:
- Marcelo will focus on the speech processing aspects of the problem (e.g., TTS, Self-Supervised Learning models).
- Enzo will investigate the specificities of EMG signals, preprocessing techniques, and standard evaluation metrics.
- Daniel will deeply analyze the STE-GAN architecture, including its internal modules, loss functions, and implementation details.

**WEEKS 4 and 5**: By this stage, we aim to have successfully reproduced the baseline STE-GAN code. This will allow us to initiate preliminary experiments, such as cross-dataset evaluations. We will also begin testing and adapting specific STE-GAN components, including the HuBERT module, the EMG Encoder, and analyzing the impact of data augmentation on the classifier.

**WEEKS 6 to 9**: In the final phase, we will solidify our proposed contributions and conduct more extensive training sessions. We will perform comprehensive evaluations across the selected datasets to compile the final metrics, analyze the results, and prepare the project's final deliverables.

## Bibliographic References

1. T. Schultz, M. Wand, T. Hueber, D. J. Krusienski, C. Herff, and J. S. Brumberg, “Biosignal-based spoken communication: A survey,” IEEE/ACM Transactions on Audio, Speech and Language
Processing, vol. 25, no. 12, pp. 2257–2271, 2017.

2. Scheck, K., Schultz, T. (2023) STE-GAN: Speech-to-Electromyography Signal Conversion using Generative Adversarial Networks. Proc. Interspeech 2023, 1174-1178, doi: 10.21437/Interspeech.2023-174

3. Wand, M., Janke, M., Schultz, T. (2014) The EMG-UKA corpus for electromyographic speech processing. Proc. Interspeech 2014, 1593-1597, doi: 10.21437/Interspeech.2014-379

4. David Gaddy and Dan Klein. 2020. Digital Voicing of Silent Speech. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5521–5530, Online. Association for Computational Linguistics.

5. Zhao, R., Bai, Y., Zhang, S. et al. An open dataset of multidimensional signals based on different speech patterns in pragmatic Mandarin. Sci Data 12, 1934 (2025). https://doi.org/10.1038/s41597-025-06213-z

6. Scheck, K., Dombeck, T., Ren, Z., Wu, P., Wand, M., Schultz, T. (2025) DiffMV-ETS: Diffusion-based Multi-Voice Electromyography-to-Speech Conversion using Speaker-Independent Speech Training Targets. Proc. Interspeech 2025, 5573-5577, doi: 10.21437/Interspeech.2025-1914

7. Zhou, D., Zhang, Y., Wu, J., Zhang, X., Xie, L., and Yin, E., “AVE Speech: A Comprehensive Multi-Modal Dataset for Speech Recognition Integrating Audio, Visual, and Electromyographic Signals”, arXiv e-prints, Art. no. arXiv:2501.16780, 2025. doi:10.48550/arXiv.2501.16780.

8. Diener, L., Vishkasougheh, M.R., Schultz, T. (2020) CSL-EMG_Array: An Open Access Corpus for EMG-to-Speech Conversion. Proc. Interspeech 2020, 3745-3749, doi: 10.21437/Interspeech.2020-2859

9. Harshavardhana T. Gowda, & Lee M. Miller. (2025). Non-invasive electromyographic speech neuroprosthesis: a geometric perspective.

10. Giusy Spacone, Sebastian Frey, Giovanni Pollo, Alessio Burrello, Daniele Jahier Pagliari, Victor Kartsch, Andrea Cossettini, & Luca Benini. (2026). SilentWear: an Ultra-Low Power Wearable System for EMG-based Silent Speech Recognition.

11. Satterlee, N.; Zuo, X.; Moon, K.; Lee, S.Q.; Peterson, M.; Kang, J.S. Sentence-Level Silent Speech Recognition Using a Wearable EMG/EEG Sensor System with AI-Driven Sensor Fusion and Language Model. Sensors 2025, 25, 6168. https://doi.org/10.3390/s25196168
