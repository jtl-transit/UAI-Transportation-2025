# AI for Transportation | Recitation 2 - Deep Learning Approaches to Discrete Choice Modeling

[**Worksheet**](./recitation_2.tex)
[**Recitation Slides**](./recitation_2_slides.pdf)
**Recitation Recording TBD**
[**Lecture Summary**](../../lectures/lecture_2.md)

<a target="_blank" href="https://colab.research.google.com/github/jtl-transit/UAI-Transportation-2025/blob/main/recitations/recitation_2/recitation_2_code.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Overview

This recitation explores advanced neural network approaches to discrete choice modeling using the **Netherlands Train Choice Dataset**. Students will implement and compare three different modeling approaches: classical Multinomial Logit (MNL), Fully Connected Deep Neural Networks (FC-DNN), and theory-informed Alternative-Specific Utility Deep Neural Networks (ASU-DNN).

The focus is on understanding the trade-offs between interpretability and predictive performance, while learning to implement theory-informed neural architectures that respect economic choice theory.

## Learning Objectives

By the end of this recitation, students will be able to:

1. **Transform and prepare choice data** from wide-format to long-format for discrete choice modeling
2. **Implement classical econometric methods** including Multinomial Logit with maximum likelihood estimation
3. **Build and train neural networks** for choice prediction using TensorFlow/Keras
4. **Design theory-informed architectures** that incorporate utility maximization theory into neural networks
5. **Conduct rigorous model evaluation** using individual-level cross-validation to prevent data leakage
6. **Interpret economic significance** of model parameters and understand utility-based choice mechanisms
7. **Compare modeling approaches** across interpretability, performance, and theoretical foundations

## Dataset: Netherlands Train Choice

The dataset contains **2,929 choice observations** from stated preference experiments where respondents chose between two train journey alternatives. Each choice scenario includes:

### Attributes per Alternative:
- **Price**: Cost in cents of Dutch guilders
- **Travel Time**: Journey duration in minutes  
- **Comfort**: Comfort level (0=High, 1=Medium, 2=Low)
- **Change**: Number of transfers/connections required

### Data Characteristics:
- **Choice scenarios**: ~1,464 unique choice situations
- **Alternatives per scenario**: 2 (binary choice)
- **Data format**: Wide-format requiring transformation to long-format
- **Choice distribution**: Nearly balanced between alternatives

## Recitation Structure

### Part 1: Data Access and Preparation 📊
**Objective**: Master choice data handling and preparation techniques

**Key Activities**:
- Load and explore the Netherlands Train dataset structure
- Understand wide vs. long format requirements for choice modeling
- Implement data transformation from wide to long format
- Conduct exploratory data analysis to understand choice patterns
- Set up individual-level cross-validation to prevent data leakage
- Analyze choice sensitivity to price, time, comfort, and transfers

**Learning Outcomes**: Students gain practical skills in choice data manipulation and understand the importance of proper data splitting in discrete choice contexts.

### Part 2: Classical Baseline - Multinomial Logit (MNL)
**Objective**: Implement and interpret traditional econometric choice models

**Key Activities**:
- Implement MNL from scratch using maximum likelihood estimation
- Understand utility theory: V = βX + ε and choice probabilities
- Estimate parameters using optimization techniques (L-BFGS-B)
- Interpret parameter signs and economic significance
- Calculate choice probabilities using softmax over utilities
- Establish baseline performance metrics for comparison

**Theory Foundation**: 
- Utility maximization: P(i) = exp(Vi) / Σ exp(Vj)
- Linear utility specification: V = β₁·price + β₂·time + β₃·comfort + β₄·change
- Economic interpretation of coefficients as marginal utilities

### Part 3: Fully Connected Deep Neural Networks (FC-DNN)
**Objective**: Explore "black box" neural approaches to choice modeling

**Key Activities**:
- Transform choice sets into flattened feature vectors [alt1_features, alt2_features]
- Design and train fully connected neural networks
- Implement feature engineering including difference variables and interactions
- Apply regularization techniques (dropout, batch normalization)
- Compare performance against MNL baseline
- Analyze prediction distributions and model behavior

**Architecture Design**:
- Input: 8-14 dimensional feature vectors (flattened alternatives + engineered features)
- Hidden layers: [128, 64, 32] with ReLU activation
- Output: Single sigmoid unit for binary choice probability
- Training: Adam optimizer with early stopping and learning rate scheduling

### Part 4: Alternative-Specific Utility DNN (ASU-DNN)
**Objective**: Implement theory-informed neural architectures that respect economic choice theory

**Key Activities**:
- Design separate neural networks for each alternative's utility function
- Implement utility-based choice probabilities: P(i) = sigmoid(Ui - Uj)
- Maintain economic interpretability through architectural constraints
- Compare alternative-specific vs. shared parameter approaches
- Evaluate benefits of theory-informed design vs. black-box methods
- Conduct comprehensive performance comparison across all models

**Theory-Informed Architecture**:
- Separate utility networks: Ui = NNi(Xi) for each alternative i
- Choice probability: P(choose alt2) = sigmoid(U₂ - U₁)
- Economic constraints: Utilities can be compared and interpreted
- Flexibility: Alternative-specific parameters allow heterogeneous preferences

## Key Technical Concepts

### Economic Theory Integration
- **Utility Maximization**: Choice follows from comparing alternative-specific utilities
- **Random Utility Models**: Incorporating both systematic and random utility components
- **Interpretability**: Maintaining economic meaning in neural network predictions

### Machine Learning Techniques
- **Feature Engineering**: Creating meaningful variables from raw attributes
- **Regularization**: Preventing overfitting through dropout and batch normalization
- **Cross-Validation**: Individual-level splitting for proper model evaluation
- **Early Stopping**: Preventing overfitting during neural network training

### Model Comparison Framework
- **Accuracy**: Prediction correctness across validation folds
- **Interpretability**: Ability to understand and explain model decisions
- **Theoretical Foundation**: Consistency with economic choice theory
- **Computational Efficiency**: Training time and model complexity

## Implementation Details

### Technical Stack
- **Python**: Core programming language
- **TensorFlow/Keras**: Neural network implementation
- **scikit-learn**: Classical ML utilities and preprocessing
- **pandas/numpy**: Data manipulation and numerical computation
- **matplotlib/seaborn**: Visualization and analysis

### Cross-Validation Strategy
- **5-fold individual-level cross-validation**
- **No data leakage**: Same individual never appears in both training and validation
- **Balanced folds**: Maintaining choice rate consistency across folds
- **Robust evaluation**: Multiple validation rounds for reliable performance estimates

## Data

We will be using the following data. However, in the CoLab notebook, we have already taken provisions for you to download them and store them in the appropriate file structure.

- **Netherlands Train Dataset:**
  - [**mlogit_Train_wide.csv:**](./data/mlogit_Train_wide.csv) 1987 Netherlands Rail vs. Car mode-choice dataset containing 2,929 revealed preference choice observations from 235 individuals across 19 choice scenarios.
  - [**Data Dictionary:**](./data/mlogit_choice_data_dictionary.pdf) Comprehensive documentation of dataset variables and structure.
  - [**Metadata:**](./data/train_metadata.pdf) Additional context about data collection methodology and survey design.

## Questions

### Section 1: Understanding the Netherlands Train Dataset

1. How does the panel structure of the data (multiple choice scenarios per individual) affect model validation strategies?
2. What are the key differences between wide format (one row per choice scenario) and long format (one row per alternative) for discrete choice modeling?
3. How do the attribute distributions (price, time, comfort, changes) vary across alternatives, and what might this tell us about the choice context?
4. Why is individual-level grouping important when splitting data for cross-validation in choice modeling?

### Section 2: Model Comparison and Performance

1. Why do the logistic regression and MNL models produce nearly identical results in the binary choice case?
2. How does the Alternative-Specific Utility (ASU) DNN architecture differ from a fully connected neural network in terms of parameter count and connectivity?
3. What role does the sparsity constraint in ASU-DNN play as implicit regularization?
4. How do the different models (MNL, FC-DNN, ASU-DNN) perform in terms of accuracy, interpretability, and parameter efficiency?

### Section 3: Economic Interpretation and Theory Integration

1. How can you extract economic measures like value of time and elasticities from neural network predictions?
2. What is the Independence of Irrelevant Alternatives (IIA) property, and how do different architectures respect or violate this assumption?
3. How does incorporating utility theory into neural architecture design affect both predictive performance and economic interpretability?
4. In what scenarios would you prefer each modeling approach (MNL vs. FC-DNN vs. ASU-DNN) for transportation planning applications?

### Section 4: Practical Implementation Considerations

1. How do the computational requirements differ between classical econometric models and neural networks for choice modeling?
2. What are the implications of limited training data versus big data scenarios for each modeling approach?
3. How might these approaches extend to more complex choice scenarios (larger choice sets, real-time data, multiple data modalities)?
4. What are the trade-offs between model interpretability and predictive flexibility in transportation planning contexts? 