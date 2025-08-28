# AI for Transportation | Recitation 3 - Multimodal Machine Learning for Transit Data

## Introduction

<a target="_blank" href="https://colab.research.google.com/github/RicoFio/UAI-Transportation-2025/blob/main/recitations/recitation_3/recitation_3_code.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

In this recitation, we examine how **multimodal machine learning** can be applied to the public transportation domain. Our aim is to see how modern models handle very different types of input, **text** and **video**, to produce actionable insights for transit operations.

We will focus on two case studies:  

1. **Text Analysis:** Applying a BERT-based language model to analyze transit agency tweets, extracting categories and identifying themes related to service quality and disruptions.  
2. **Video Analysis:** Using semantic segmentation on station footage to estimate passenger density, providing a proxy for crowding and flow management.  

By working through these examples, you will learn how to build **end-to-end inference pipelines** for both language and vision data, develop intuition about their strengths and limitations, and reflect on how such approaches could be extended in practice and research to improve transit systems.  

## Data

We will be using the following data. However, in the CoLab notebook, we have already taken provisions for you to download them and store them in the appropriate file structure.

- **Natural Language Processing**
  - [**Twitter Data Sample:**](https://drive.google.com/file/d/1ucmdRh5s_KuWxOWdfZhbcNL2ACtedO85/view?usp=drive_link) Sample of 5000 tweets directed to or mentioning the Washington Metro Area Transit Authority (WMATA).
- **Computer Vision**
  - [**Vienna Metro Video Sample:**](https://drive.google.com/file/d/1Mj8nESD2IcqoqJen_X9dhJlTn3iaVPe2/view?usp=drive_link)
