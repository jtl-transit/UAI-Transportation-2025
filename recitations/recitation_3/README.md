# AI for Transportation | Recitation 3 - Multimodal Machine Learning for Transit Data

[**Homework**](../../homeworks/homework_3.pdf)
[**Recitation Slides**](./recitation_3_slides.pdf)
[**Lecture**](../../lectures/lecture_3.md)

<a target="_blank" href="https://colab.research.google.com/github/jtl-transit/UAI-Transportation-2025/blob/main/recitations/recitation_3/recitation_3_code.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Introduction

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

## Questions

### Section 1: Using BERT on Transit Tweets 

1. In the example above, we used MetroBERTa to analyze the sentiment of transit tweets. Which of the following would you also be able to use sentiment analysis for?
2. Why do we have to clean the raw Twitter/X data? Which function above is designed to clean and tokenize?
3. How did sentiment of the Washington Metro change over time in late 2024?
4. What can we infer from the sentiment analysis, and are there any other insights you can gain from the data we enriched?


### Section 2: Using BERT on Transit Tweets 

1. If you see increased platform crowding during specific events (eg. a concert, parade, etc.), how might that inform scheduling?
2. Where can we also aquire actionable video data?
3. What are some other examples of information you can get from transit surveillance data?
4. If you were a transit operator, how could this type of pixel coverage analysis inform your decisions during peak hours?  
5. Why might transfer learning work well in some cases but fail in others?
6. Can you think of examples outside of transit where this issue could appear?