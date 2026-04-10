# EEGAN - Da fala ao EMG

# EEGAN - From speech to EMG

## Presentation

This project originated in the context of the graduate course _IA376N - Generative AI: from models to multimodal applications_,
offered in the first semester of 2026, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> |Name | RA | Specialization|
> |--|--|--|
> | Daniel Neto | 169408 | Computer Engineering|
> | Enzo | 123456 | Computer Engineering|
> | Marcelo | 123456 | Computer Engineering|

## Project Summary Description

> Description of the project theme, including generating context and motivation.

This project is in the context of biological signals. These signals are of difficult acquisition, because subjects must undergo a tiresome procedure to create a small amount of data; in most cases, data collected is lawfully restricted, and the number of subjects willing to undergo such a procedure is minute (resulting in low variability). Thus, there is a strong motivation for the synthethic generation of such signals.

A type of biological signal for which this is specially true is Surface Electromyography (EMG) in the context of speech, where articulatory muscles reflect the audio signal production. These biosignals are of interest for Silent Speech Interfaces (SSIs) [1], which aim to enable speech communication without depending on acoustic speech.

To address the issues stated above, as well as to develop part of a SSI, we propose a method of generating EMG signals, based on the work of [2].

> Description of the main goal of the project.

The main goal of the project is to generate reliable EMG data, which is not only similar to the target domain, but is also capable of retaining high accuracy scores (by the Word Error Rate metric) after being converted to audio.

> Clarify what the output of the generative model will be.

The output of the generative model will be EMG signals of the same dimension as the input.

> Include in this section a link to the presentation video of the project proposal (maximum 5 minutes).

INCLUDE LINK

## Proposed Methodology

> - Which dataset(s) the project intends to use, justifying the choice(s).

For this model, the datasets considered will all contain paired EMG/Audio signals. 

Since this type of data is rare, these datasets compose the domain of signals that is publicly available.

They are:

- The EMG-UKA corpus for electromyographic speech processing [3]
- Digital Voicing of Silent Speech [4]
- An open dataset of multidimensional signals based on different speech patterns in pragmatic Mandarin [5]
- DiffMV-ETS: Diffusion-based Multi-Voice Electromyography-to-Speech Conversion using Speaker-Independent Speech Training Targets [6]
- AVE Speech Dataset: A Comprehensive Benchmark for Multi-Modal Speech Recognition Integrating Audio, Visual, and Electromyographic Signals [7]
- CSL-EMG_Array: An Open Access Corpus for EMG-to-Speech Conversion [8]
- emg2speech: synthesizing speech from electromyography using self-supervised speech models [9]
- SilentWear: an Ultra-Low Power Wearable System for EMG-based Silent Speech Recognition [10]
- Sentence-Level Silent Speech Recognition Using a Wearable EMG/EEG Sensor System with AI-Driven Sensor Fusion and Language Model [11]

> - Which generative modeling approaches the group already sees as interesting to be studied.

The generative modeling approach that will be the base for this study is the GAN presented in [2]. We found it to be interesting because it obtained a significant result for the time and for introducing a novelty regarding GANS: instead of naive sampling from a Gaussian in the Z latent space, it sampled from a known, controllable latent space, that is, an audio signal.

> - Reference articles already identified and that will be studied or used as part of the project planning.

Reference articles are cited in the bibliography. Other bibliography will be added throughout the development of the project.

> - Tools to be used (based on the group’s current vision of the project).

PyTorch, Github

> - Expected results.

We expect the greater variation introduced by new components and datasets to improve the evaluation scores of [2].

> - Proposal for evaluating the synthesis results.

The quality of the results will be measured by the proximity of: the synthetic data to samples from the target distribution (EMG/EMG), as well as the synthetic data features (Speech Units, Phonemes) to the ground truth target features.

The synthetic data will also be converted to audio so that the WER accuracy score can me measured.

## Schedule

> Proposed schedule. Try to estimate how many weeks will be spent on each stage of the project.

WEEK 1 to 3: Introductory study about main problem, including context, model, and datasets evaluations
- One dataset per person, per week (9 total)
- Marcelo: Speech (TTS, SSL)
- Enzo: Details about EMG (related works, etc.)
- Daniel: Models details (STE-GAN, loss, operations, etc.)

WEEK 4 to 6: Experimentations of limited scope: HuBERT, EMG Encoder, Cross Dataset Evaluation, Augmentation impact on classifier

WEEK 6 to 9: Definition of contribution results, training and models evaluations

## Bibliographic References

> Point out in this section the bibliographic references adopted in the project.

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
