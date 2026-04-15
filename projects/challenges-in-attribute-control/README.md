# `Geração de Imagens com IA: Desafios no Controle de Atributos sob Viés de Dados`
(Sugestão: Geração de Imagens com IA: Diversidade, Imparcialidade e Confiabilidade dos Modelos Generativos)

# `AI Image Generation: Challenges in Attribute Control under Data Bias`
(Sugestão: AI Image Generation: Diversity, Fairness, and Reliability of Generative Models)

## Presentation

This project originated in the context of the graduate course _IA376N - Generative AI: from models to multimodal applications_, offered in the first semester of 2026, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

Link to the presentation slides: https://canva.link/a1xy936n5saa099

> |Name | RA | Specialization|
> |--|--|--|
> | Patrícia P. Giordano | 971352 | Computer Engineering|
> | Gabriel Morais Alves | 323616 | Computer Engineering|
> | Silvia A. P. Olivio | 224932 | Electrical Engineering|

## Project Summary Description

### Description of the project theme, including generating context and motivation.

This project aims to evaluate the behavior of text-to-image generative AI models, focusing on their ability to correctly interpret simple textual descriptions. The rapid advancement of generative models has enabled high-quality image synthesis from text; however, these models often exhibit limitations related to reliability, bias, and control over specific attributes.
(This project aims to evaluate the behavior of text-to-image generative AI models, focusing on their ability to correctly interpret simple textual descriptions. The rapid advancement of generative models has enabled high-quality image synthesis from text; however, these models often exhibit limitations related to diversity, fairness, and reliability.)

The main goal of this project is to investigate how pre-trained open-source models, such as Stable Diffusion, available through platforms like Hugging Face, handle prompts where there is a potential 'limitation' between the user’s input and the statistical patterns learned during training.The project is motivated by the observation that these models frequently fail to correctly apply specific attributes—such as color—to objects. For example, prompts such as:
- “white carrot”,
- “pink classroom blackboard” and
- “purple polar bear”.
  
Often result in images where the model ignores the requested attribute and instead generates outputs aligned with common real-world representations, such as orange carrots, green/black boards, white polar bears, as the examples below:

![Resultado do experimento](imagens/Pink_board_2.jpg)
![Resultado do experimento](imagens/orange_purple.jpeg)
![Resultado do experimento](imagens/yellow_cloud.jpeg)

This behavior suggests that the model prioritizes learned statistical correlations over explicit user instructions, revealing a limitation in its ability to disentangle attributes, e.g., color, from object identity.

The central research question of this project is:

> #### *Is this limitation primarily caused by biases in the training data distribution, or by constraints in the model architecture itself?*
(Is this limitation primarily caused by the training data or by constraints in the model's architecture itself?)


## Main Goal

The main objective of this project is to evaluate whether the application of fine-tuning techniques to pre-trained open-source text-to-image generation models — Stable Diffusion — improves the fidelity in representing specific visual attributes, such as color, with respect to the instructions provided in the prompt.

Specifically, the study aims to investigate whether, after fine-tuning with datasets that present greater diversity of these attributes, e.g. different colors applied to the same object, the model is able to correctly apply the requested color to specific regions of the image (such as a classroom blackboard), or whether it still exhibits limitations in attribute localization and control, resulting in the incorrect application of color to other regions of the scene (such as walls or adjacent objects), especially in elements that exhibit strong bias in the training data, such as classroom blackboards, which are traditionally associated with green or black colors.
(Specifically, the study aims to investigate whether, after fine-tuning with datasets that present greater diversity of these attributes, e.g. different colors applied to the same object, the model is able to correctly apply the requested command in the prompt or whether it still exhibits limitations, resulting in an incorrect response to the user's request in the prompt.)

# Main Hypothesis
The main hypothesis of this project is that the inability of text-to-image models to correctly apply specific attributes is primarily influenced by biases in the training data distribution, which leads the model to favor statistically dominant representations over explicit user instructions.
(The main hypothesis of this project is that the inability of text-to-image models to correctly apply specific attributes is primarily influenced by the statistical prioritization of patterns, which leads the model to favor dominant representations over explicit user instructions, such as a classroom blackboard traditionally associated with green or black colors.)

# Secondary Questions 
- To support the main hypothesis, the project investigates the following secondary questions:
- To what extent do training data biases influence the model’s output when attributes conflict with common representations?
- Can fine-tuning with controlled data improve the model’s ability to correctly apply attributes?
- Does the model architecture itself limit the ability to disentangle attributes from objects?
- Are alternative architectures (e.g., Beta-VAE) more effective in handling attribute control?

# Expected Output of the Generative Model.
The output of the generative model in this project will consist of images generated from controlled textual prompts, with a specific focus on attribute manipulation, especially color.

In the initial state (pre-trained model), the system is expected to generate images influenced by learned statistical patterns, often failing to correctly apply uncommon attributes specified in the prompt.

After applying fine-tuning with a synthetic dataset composed of objects with non-standard attribute combinations (e.g., unusual colors), the expected state-of-the-art outcome is that the model becomes capable of correctly modifying object attributes according to the prompt specification.
(After applying fine-tuning to a synthetic dataset, including combinations of non-standard attributes, such as unusual colors, the expected outcome is that the model be able of correctly modifying object attributes according to the prompt's requests.)

In particular, the improved model should be able to:

- Correctly apply specified attributes, such as color, to objects, even when they contradict common real-world representations
- Cuidado: Reduce bias toward dominant patterns learned during pre-training
- Generate images that are more consistent with the input prompt

## Proposed Methodology

# Dataset
Generate a synthetic dataset with non-standardized object attributes, e.g., colored blackboards, and use LoRA fine-tuning to improve attribute control in a pre-trained text-to-image model.

## Which dataset(s) the project intends to use, justifying the choice(s).
A synthetic dataset synthesized by the group containing non-standardized object attributes, e.g., colored blackboards. Our choice for this type of data was due to the context of the problem, in which is very specific for some type of attributes, which makes it difficult for us to encounter existent datasets containing exactly the type of attribute we need.

## Which generative modeling approaches the group already sees as interesting to be studied.
The following methods are considered within the scope of this project:
- Diffusion-based models, particularly Latent Diffusion Models used in Stable Diffusion, which serve as the primary architecture for image generation.
- LoRA (Low-Rank Adaptation), which will be used as an efficient approach for fine-tuning the pre-trained model with controlled data.

**If the fine-tuned dataset does not yield satisfactory results and sufficient time is available, VAE-based models may be explored, with a focus on analyzing attribute disentanglement. However, this step is considered optional and may be addressed as future work, depending on time constraints and overall project scope.**

## Reference articles already identified and that will be studied or used as part of the project planning.

- Rombach, R. et al. (2021). “High-Resolution Image Synthesis with Latent Diffusion
Models”. Em: CoRR abs/2112.10752. url: https://arxiv.org/abs/2112.10752 (ver p. 20).

- Ho, J., A. Jain e P. Abbeel (2020). “Denoising Diffusion Probabilistic Models”. Em: CoRR
abs/2006.11239. url: https://arxiv.org/abs/2006.11239 (ver p. 20).

- Nichol, A. et al. (2021). “GLIDE: Towards Photorealistic Image Generation and Editing
with Text-Guided Diffusion Models”. Em: CoRR abs/2112.10741. url: https://arxiv.org/abs/2112.10741 (ver p. 20).

## Tools to be used (based on the group’s current vision of the project).
- Python
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion
- LoRA (Low-Rank Adaptation)
- CLIP (for evaluation)
- Google Colab (for training and experiments)

## Expected results.
We expect to verify that the fine-tuning techniques were indeed helpful into adjusting the addressed problem, and that will be verified using the proposed evaluation metrics:
- CLIP - measurement of similarity (prompt, image)
- Attribute Binding Evaluation
- VQA-based evaluation - usage of vision language model to answer questions and compare with expected results
- Human evaluation - realism and subtle failures
- Compositional benchmarks - success rate of correct binding

## Proposal for evaluating the synthesis results.
CLIP Score: measuring the semantic alignment between generated images and input prompts

## Schedule

The following schedule is proposed for each stage of the project:

![Schedule - IA Generative Project](https://github.com/user-attachments/assets/90eafcc8-3f88-4fc1-b792-126e120d7ae2)


## Bibliographic References

- Vice, J., Akhtar, N., Sigal, L. et al. On the fairness, diversity and reliability of text-to-image generative models. Artif Intell Rev 59, 57 (2026). https://doi.org/10.1007/s10462-025-11424-2
