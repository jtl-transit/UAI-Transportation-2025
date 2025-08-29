# AI for Transportation | Recitation 4 - Generative AI for Urban Planning

[**Homework**](../../homeworks/homework_4.pdf)
[**Recitation Slides**](./recitation_4_slides.pdf)
[**Lecture**](../../lectures/lecture_4.md)

<a target="_blank" href="https://colab.research.google.com/github/jtl-transit/UAI-Transportation-2025/blob/main/recitations/recitation_4/recitation_4_code.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Introduction

In this recitation, we explore how conditional diffusion models, specifically **ControlNet**, can be used to generate urban imagery under structured constraints. Our goal is to see how modern generative AI models can combine **text prompts** with **spatial condition images** to synthesize layouts that are both realistic and aligned with planning objectives.  

We will walk through three main steps:  

1. **Model Setup:** Loading pretrained ControlNet weights and initializing the inference pipeline.  
2. **Conditioned Generation:** Using a control image (e.g., a land-use or road network map) alongside a textual description to guide the diffusion process.  
3. **Evaluation:** Adjusting key parameters such as diffusion steps, guidance scale, and control strength, and comparing generated outputs with ground truth imagery.  

Through this exercise, you will see how generative AI pipelines are built for urban planning tasks, develop intuition about the trade-offs between realism and adherence to constraints, and reflect on possible applications for design, analysis, and research.  

## Data

We will be using the following data. However, in the CoLab notebook, we have already taken provisions for you to download them and store them in the appropriate file structure.

- [**Hint TIF:**](https://drive.google.com/open?id=1JYzXbUbhspfPB9B4IpSKjUXYSrH8Bq6Z&usp=drive_copy) Conditioning input to set hard constraints on the diffusion process of the Urban ControlNet.
- [**True TIF:**](https://drive.google.com/open?id=1xv3bfbdl9gtKDJ8XvS3m5zGi8Zn1-uAz&usp=drive_copy) Ground-truth data, following the `Hint TIF` and matching with the prompt in `recitation_4_code.ipynb`.
- [**Model checkpoint:**](https://drive.google.com/open?id=1hO409B3i3kOWAspsSIfof7FogSnk1RNX&usp=drive_copy) Training checkpoint of model, including weights.
- [**Model configuration:**](https://drive.google.com/open?id=1Fgb3Cofs6Mc9mNqXk1BW9Io5Iud2V8un&usp=drive_copy) Information on model architecture and computational pipeline.

## Preparation

In the `./code/` folder, you can find the scripts we have used to [*quantize* the model](https://huggingface.co/docs/optimum/en/concept_guides/quantization), i.e. reduce the precision of the weight values from a 64-bit float to a 16-bit float, allowing us to have more light-weight models with similar performance.
