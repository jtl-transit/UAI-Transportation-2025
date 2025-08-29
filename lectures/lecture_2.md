# Lecture 2: Models and Approaches

Lecture links:

- [**Section 1:** Background]()
- [**Section 2:** Extracting Economic Information for Interpretation]()
- [**Section 3:** Architectural Design with Alternative-Specific Utility Functions]()
- [**Section 4:** Theory-based Residual Neural Network]()
- [**Section 5:** Deep Hybrid model with Urban Imagery]()

# Automatic Transcript

## Section 1: Background
**Topic:** Methods and Models – Discrete Choice Models and Deep Neural Networks  
### 1. Introduction
- Focus: **methods and models** in AI for transportation.  
- Classic tool: **Discrete Choice Models (DCM)**.  
- Modern backbone: **Deep Neural Networks (DNNs)**.  
- Goal: Show how **DCMs** and **DNNs** can be integrated.  
#### Lecture roadmap (4 sections):
1. Extracting economic information from trained DNNs.  
2. Using DCM utility functions to inform DNN architecture.  
3. Combining DNNs with DCMs through **theory-based residual networks**.  
4. Building **deep hybrid models** to integrate **urban imagery** into transportation mode analysis.  

### 2. Background: Predicting Individual Choice
#### Example: Car vs. Bus
- **Bus**: $4, 40 minutes.  
- **Car**: $10, 25 minutes.  
- Task: predict probability of choosing each mode.  
#### Other applications:
- Travel mode choice: walk, cycle, bus, car.  
- Ride-hailing: solo vs. shared Uber.  
- Housing location, job choice, car ownership, EV adoption.  

### 3. Aggregate vs. Individual Analysis
- **Aggregate**: e.g., number of taxi trips in a neighborhood.  
- **Individual**: e.g., “Do I take a taxi at 7:00 to Central Park?”  
- Note: aggregate = sum of individuals; but not vice versa.  

### 4. Discrete Choice Model (DCM) Framework
Four components:  
1. **Decision maker** – person, household, firm.  
2. **Decision-maker characteristics** – age, gender, income, education.  
3. **Choices** – e.g., bus, car, walk, cycle.  
4. **Attributes of alternatives** – cost, time, reliability.  
5. **Decision rule** – utility maximization.  

### 5. Mathematical Formulation
- **Utility of alternative i for person n**:  
  \[
  U_{ni} = V_{ni} + \epsilon_{ni}
  \]  
  where:  
  - \(V_{ni}\) = deterministic utility  
  - \(\epsilon_{ni}\) = random error  

- **Deterministic utility**:  
  \[
  V_{ni} = \beta_1 \cdot \text{time} + \beta_2 \cdot \text{cost} + \beta_3 \cdot \text{reliability} + \dots
  \]

- **Choice probability**:  
  \[
  P_{ni} = \Pr\left(U_{ni} = \max_j U_{nj}\right)
  \]

- With **extreme value distribution assumption**:  
  \[
  P_{ni} = \frac{e^{V_{ni}}}{\sum_{j \in C_n} e^{V_{nj}}}
  \]

- This is the **Multinomial Logit (MNL) Model**.  

### 6. Extensions of DCM
- **Generalized Random Utility Model (GRUM)**.  
- Incorporates **latent class** and **latent variable models**.  
- Expands representation of unobserved heterogeneity.  

### 7. Deep Neural Networks (DNNs) for Choice Analysis
- Treat choice prediction as a **classification problem**.  
- **Inputs (X):** attributes, socio-demographics.  
- **Outputs (Y):** chosen alternative.  
- **Architecture:** multiple hidden layers → choice probabilities.  
- Alternatives: discriminant analysis, Bayesian models, SVM, KNN, decision trees, random forests.  
- This lecture focuses on **DNNs exclusively**.  

### 8. Connecting DCM and DNN
- **Observation:** MNL ≈ special case of DNN with **softmax activation**.  
- **Differences:**
  - MNL: shallow (no hidden layers), sparse connections.  
  - DNN: deep, fully connected.  

- **Key insight:**  
  - DCM = interpretable, theory-driven.  
  - DNN = flexible, data-driven.  
  - Together: potential for **hybrid, interpretable AI models**.  


## Section 2: Extracting Economic Information for Interpretation
### 1. Objective
- Use **Deep Neural Networks (DNNs)** to predict travel choices (e.g., car vs. bus).  
- Go beyond prediction: **extract economic information** that transportation professionals rely on.  
- Key question: How do we translate DNN outputs/weights into **meaningful economic measures**?  

### 2. Economic Information of Interest
Transportation professionals typically care about:  
- **Choice probability** – e.g., 40% chance to choose car.  
- **Choice prediction** – which mode a person will choose.  
- **Market share** – aggregate adoption rates (e.g., bus share in Boston).  
- **Substitution patterns** – e.g., Uber vs. taxi substitutability.  
- **Elasticity** – sensitivity of demand to price changes.  
  - Example: 10% fare increase in Uber → how much demand decreases.  
- **Marginal Rate of Substitution (MRS)** – trade-off between attributes.  
  - Example: travel time vs. travel cost → monetary **value of time**.  
- **Social welfare measures** – changes in consumer surplus from policy or price shifts.  

### 3. Methodology
- **Inputs:** attributes of alternatives + decision-maker characteristics.  
- **Model:** fully connected DNN outputs choice probabilities.  
- **Post-model analysis:**  
  - Use DNN outputs to compute the above economic quantities.  
  - DNN treated as a **black box**, then interpret results afterward.  

### 4. Example: Substitution Patterns
- **Setup:** five alternatives – walking, public transit, ride-hailing, driving, transit vehicle.  
- **Analysis:** vary cost of driving → observe change in mode probabilities.  
#### Results:
- **DNN (left plot):**  
  - As driving cost increases → car choice probability declines.  
  - Other modes (walking, transit, etc.) increase.  
  - Captures nuanced, complex behavior patterns.  
  - Harder to interpret due to irregular patterns.  

- **MNL (right plot):**  
  - Produces smooth, clean functional relationship.  
  - Easier to interpret, less flexible.  

### 5. Key Insights
- **DNNs** can recover standard economic measures post-estimation.  
- **MNL models**: simpler, interpretable, smooth patterns.  
- **DNN models**: richer, nuanced, but more complex and harder to interpret.  
- **Trade-off:**  
  - Use DNNs for flexibility and richer behavior.  
  - Use MNL for transparency and interpretability.  


## Section 3 : Architectural Design with Alternative-Specific Utility Functions
### 1. Objective
- Use **discrete choice model (DCM) theory** to guide the **architecture design** of deep neural networks (DNNs).  
- Focus: **Alternative-Specific Utility (ASU)** representation.  
- Goal: make DNNs both **efficient** and **interpretable**.  

### 2. Background
- Architectural design is critical for DNN performance.  
- Classic innovations: LeNet (1998), AlexNet (2012), ResNet, etc.  
- Limitation: DNN architectures are often **not interpretable**.  
- Idea: borrow **utility specification theory** from DCMs to inform DNN design.  

### 3. Model Architectures
#### (a) Fully Connected DNN (Vanilla)
- All attributes of all alternatives connected to all choice probabilities.  
- Dense weight matrix → everything influences everything.  
#### (b) Alternative-Specific Utility (ASU) DNN
- Attributes of each mode only connected to that mode’s probability.  
  - Example: car attributes (time, cost, reliability) → only affect probability of car.  
  - Bus attributes → only affect probability of bus.  
- Structure = **sparse** connections (diagonal matrix).  
- Acts as **implicit regularization** by enforcing sparsity.  
#### (c) Group-Based DNN
- Alternatives grouped; attributes only connected within groups.  
- Example:  
  - Group 1: alternatives 1 & 2 fully connected.  
  - Group 2: alternatives 3, 4, 5 fully connected.  
- Produces **block matrix** connectivity.  

### 4. Efficiency Implications
- ASU DNN reduces estimation error.  
- Efficiency gain can be expressed as:

\[
\text{Var}(\hat{\theta}_{\text{ASU}}) \approx \frac{1}{K^{D/2}} \, \text{Var}(\hat{\theta}_{\text{FC}})
\]

where:  
- \(K\) = number of alternatives,  
- \(D\) = dimensionality of attributes,  
- \(\hat{\theta}_{\text{ASU}}\) = estimator under ASU DNN,  
- \(\hat{\theta}_{\text{FC}}\) = estimator under fully connected DNN.  

- Interpretation: The more alternatives, the greater the efficiency benefit.  

### 5. IIA Constraint (Independence of Irrelevant Alternatives)
#### Classic MNL context:
- Ratio of probabilities between two modes depends only on their attributes:  

\[
\frac{P_1}{P_2} = f(X_1, X_2)
\]

- Irrelevant alternatives (e.g., walking) do not affect the ratio.  
#### Mapping architectures:
- **ASU DNN**: Respects IIA (like MNL).  
- **Group-Based DNN**: Similar to **nested logit**; IIA holds within groups, not across.  
- **Fully Connected DNN**: Violates IIA (like mixed logit).  

### 6. Empirical Evaluation
#### Datasets:
- **Singapore (SGP)**: ~8,000 samples.  
- **Trends model**: ~3,000 samples.  
- Models tested: ASU DNN vs. Fully Connected DNN.  
#### Results:
1. **Prediction Accuracy**  
   - Both ASU DNN and fully connected DNN outperform other classifiers.  
   - ASU DNN consistently outperforms fully connected DNN on validation and test sets.  

2. **Regularization Effectiveness**  
   - ASU DNN = implicit regularization via sparsity.  
   - Outperforms generic-purpose regularization methods.  
   - Shown by higher accuracy curves across different regularization levels.  

3. **Interpretability (Substitution Patterns)**  
   - Fully Connected DNN: nuanced, complex, but hard to interpret.  
   - Classical MNL: smooth, clean curves, but overly simplistic.  
   - ASU DNN: balance between complexity and interpretability.  
     - Captures realistic patterns while remaining constrained and interpretable.  

### 7. Summary
- **ASU DNN** introduces domain knowledge (DCM utility structure) into DNN architecture.  
- Benefits:  
  - Higher predictive performance.  
  - Implicit regularization through sparsity.  
  - Improved interpretability of substitution patterns.  
- Conceptual mapping:  
  - **ASU DNN ↔ Multinomial Logit (MNL)**.  
  - **Group-Based DNN ↔ Nested Logit**.  
  - **Fully Connected DNN ↔ Mixed Logit**.  
- Overall: **domain knowledge–driven architecture design** makes DNNs more efficient and interpretable.  


## Section 4: Theory-based Residual Neural Network

### 1. Objective
- Introduce the **Theory-Based Residual Neural Network (TB-ResNet)**.  
- Aim: combine **Discrete Choice Models (DCM)** and **Deep Neural Networks (DNNs)** in a principled way.  
- Inspiration: **ResNet** architecture with skip connections.  

### 2. Conceptual Design
- **Classic ResNet**:  
  - Two pathways from input to output:  
    1. Deep pathway through hidden layers.  
    2. Shortcut connection (identity mapping) from input to output.  

- **TB-ResNet**:  
  - Replace identity mapping with a **utility function from a DCM** (e.g., multinomial logit).  
  - Final utility is a **weighted average** of DCM utility and DNN utility:  

\[
U(x) = \delta \, V_{\text{DCM}}(x) + (1-\delta) \, V_{\text{DNN}}(x)
\]

where:  
- \(V_{\text{DCM}}(x)\) = deterministic utility from discrete choice model,  
- \(V_{\text{DNN}}(x)\) = utility from neural network,  
- \(\delta \in [0,1]\) = mixing parameter.  

### 3. Training Procedure
1. **Stage 1**: Estimate discrete choice model → obtain \(V_{\text{DCM}}(x)\).  
2. **Stage 2**: Plug \(V_{\text{DCM}}(x)\) into TB-ResNet as skip connection.  
3. Learn DNN parameters jointly with weight \(\delta\).  

- Flexible:  
  - Any DCM can serve as \(V_{\text{DCM}}\).  
  - Any DNN architecture can serve as \(V_{\text{DNN}}\).  

### 4. Application Scenarios
Tested TB-ResNet in three contexts:  
1. **Classic choice modeling** – mode choice (car, bus, walk, etc.).  
2. **Prospect theory** – choices between risky outcomes.  
3. **Hyperbolic discounting** – intertemporal choices (\(X_1, T_1\) vs. \(X_2, T_2\)).  

### 5. Evaluation Criteria
Three dimensions:  
1. **Prediction accuracy**.  
2. **Interpretability of utility functions**.  
3. **Robustness to noise/adversarial attacks**.  

### 6. Results
#### (a) Prediction Accuracy
- Extreme cases:  
  - \(\delta = 1 \implies\) pure DNN.  
  - \(\delta = 0 \implies\) pure DCM.  
- Optimal accuracy achieved **in between** (\(0 < \delta < 1\)).  
- Pattern: inverted U-shape for accuracy vs. \(\delta\).  
- Maximum prediction accuracy depends on scenario (choice, prospect, or discounting).  

#### (b) Interpretability
- **MNL (pure DCM):**  
  - Smooth, clean relationships (e.g., bus probability ↓ with cost/time ↑).  
  - Easy to interpret, but oversimplified.  

- **Pure DNN:**  
  - Captures nuanced, nonlinear patterns.  
  - Hard to interpret.  

- **TB-ResNet (mixture):**  
  - Combines both advantages:  
    - Structured global pattern from DCM.  
    - Local nuances from DNN.  
  - Best balance achieved at intermediate \(\delta\).  

#### (c) Robustness
- Tested with:  
  - Fast Gradient Sign Method (FGSM) attack.  
  - Target Gradient Sign Method attack.  
  - Gaussian noise perturbations.  

- **Findings:**  
  - **MNL (DCM):** most robust, stable under noise.  
  - **DNN:** least robust, accuracy drops quickly under attack.  
  - **TB-ResNet:** more robust than pure DNN, though not as robust as MNL.  
  - Trade-off: robustness vs. predictive power.  

### 7. Summary
- **TB-ResNet** extends ResNet by embedding **theory-based utility functions** in the skip connection.  
- Key features:  
  - **Higher prediction accuracy**: better than DCM, marginally better than DNN.  
  - **Improved interpretability**: structured global trend + nuanced local behavior.  
  - **Better robustness**: more stable than DNN under adversarial noise.  

- Conceptual advantages:  
  - Simple, generic, and flexible.  
  - Can combine any DCM with any DNN.  
  - Provides a **principled hybrid** of theory and machine learning.  
#### (c) Robustness
- Tested with:  
  - Fast Gradient Sign Method (FGSM) attack.  
  - Target Gradient Sign Method attack.  
  - Gaussian noise perturbations.  

- **Findings:**  
  - **MNL (DCM):** most robust, stable under noise.  
  - **DNN:** least robust, accuracy drops quickly under attack.  
  - **TB-ResNet:** more robust than pure DNN, though not as robust as MNL.  
  - Trade-off: robustness vs. predictive power.  

### 7. Summary
- **TB-ResNet** extends ResNet by embedding **theory-based utility functions** in the skip connection.  
- Key features:  
  - **Higher prediction accuracy**: better than DCM, marginally better than DNN.  
  - **Improved interpretability**: structured global trend + nuanced local behavior.  
  - **Better robustness**: more stable than DNN under adversarial noise.  

- Conceptual advantages:  
  - Simple, generic, and flexible.  
  - Can combine any DCM with any DNN.  
  - Provides a **principled hybrid** of theory and machine learning.  

## Section 5: Deep Hybrid model with Urban Imagery

### 1. Objective
- Explore how **deep hybrid models** can integrate **structured data** (socioeconomic, travel survey data) and **unstructured data** (urban imagery).  
- Goal:  
  1. Improve travel behavior analysis.  
  2. Attach **economic interpretation** to urban imagery.  

### 2. Motivation
- **Classic demand modeling**:  
  - Handles structured numerical data (e.g., travel time, cost, demographics).  
  - Cannot deal with unstructured inputs (images, text, audio, video).  

- **Deep learning**:  
  - Naturally handles unstructured data.  
  - Less interpretable, but highly flexible.  

- **Images in decision-making**:  
  - Built environments, neighborhood visual cues influence travel and consumption decisions.  
  - Integrating imagery into demand models adds **behavioral realism**.  

### 3. Deep Hybrid Model Architecture

- **Vertical path (classic demand modeling):**  
  \[
  X \;\; \rightarrow \;\; Z \;\; \rightarrow \;\; Y
  \]
  - \(X\): socioeconomic variables (low-dim, ~10).  
  - \(Z\): latent utility space.  
  - \(Y\): travel behavior outcome (e.g., mode choice, car ownership, departure time).  

- **Horizontal path (urban imagery):**  
  \[
  I \;\; \xrightarrow{\text{Encoder}} \;\; Z \;\; \xrightarrow{\text{Decoder}} \;\; \hat{I}
  \]
  - \(I\): urban imagery (high-dim, ~2048).  
  - Encoded to latent space \(Z\), reconstructed via decoder.  

- **Mixing operator:**  
  \[
  Z = M(X, I) = \lambda X + (1-\lambda) I, \quad \lambda \in [0,1]
  \]
  - Controls mixture between structured and unstructured data.  

### 4. Case Study: Chicago
- **Data sources:**  
  - Travel survey: structured socioeconomic data.  
  - Satellite imagery: built environment of neighborhoods.  

- **Dimensions:**  
  - \(X\): low-dim socioeconomic features (~10).  
  - \(I\): high-dim imagery features (~2048).  
  - \(Y\): low-dim travel behavior outcomes.  

### 5. Results
#### (a) Navigating the Imagery Space
- Built environment can be **morphed** between neighborhoods:  
  - South Chicago → North Chicago.  
  - South Chicago → Central Chicago.  
- Latent space \(Z\) represents gradual transitions of urban form.  
#### (b) Economic Interpretation of Imagery
- **Social welfare mapping:**  
  - Moving from South → North or South → Central Chicago increases welfare.  

- **Mode choice probabilities:**  
  - Transition South → North Chicago:  
    - Car use ↑ (higher incomes, more car ownership).  
    - Walking/public transit ↓.  
  - Transition South → Central Chicago:  
    - Public transit ↑ (better downtown service).  
    - Car use ↓.  

### 6. Implications
#### Demand Modeling
- Integrates **urban imagery** with numerical data.  
- Improves prediction accuracy.  
- Adds **behavioral realism** to choice models.  
- Enables **economic interpretation of images** (e.g., value of built environment changes).  
#### Urban Planning
- Generates **hybrid built environments**:  
  - Simulates transitions between neighborhoods.  
  - Associates each generated environment with **social welfare** and **mode share**.  
- Supports socially aware urban design by linking imagery to outcomes.  
### 7. Broader Summary of Lecture 2
- Four examples of combining **DNN** and **DCM**:  
  1. **Post-model analysis**: extract economic measures from DNNs.  
  2. **Architecture design**: use alternative-specific utility functions for sparse, interpretable DNNs.  
  3. **Theory-based ResNet**: hybrid weighted average of DCM and DNN utilities.  
  4. **Deep hybrid model**: integrate unstructured imagery with structured data.  

### 8. Theoretical Insight: Approximation vs. Estimation Error
- **Prediction error = Approximation error + Estimation error**.  
  - **Approximation error** ↓ with more complex models.  
  - **Estimation error** ↑ with more complex models (requires more data).  

- **Position of models:**  
  - **DCM (e.g., MNL):** simple, high approximation error, low estimation error.  
  - **DNN:** highly expressive, low approximation error, high estimation error.  

- **Hybrid goal:**  
  - Enrich DCMs (reduce approximation error).  
  - Regularize/simplify DNNs (reduce estimation error).  

### 9. Key Challenges and Solutions
1. **Sensitivity to hyperparameters**  
   - Mitigation: regularization, domain-knowledge-based architectures (ASU DNN, TB-ResNet).  

2. **Model non-identification**  
   - Insight: local optima may be sufficient for practical modeling.  

3. **Local irregularity (robustness issues)**  
   - Mitigation: hybrid designs improve robustness (e.g., TB-ResNet under adversarial noise).  

### 10. Final Takeaway
- **Discrete Choice Models (DCM)** = interpretable, theory-based, robust.  
- **Deep Neural Networks (DNN)** = flexible, powerful, data-driven.  
- **Hybrid models** combine strengths:  
  - Higher accuracy.  
  - Greater interpretability.  
  - Better robustness.  
- **Conclusion of Lecture 2:**  
  - Future of AI in transportation lies in **bridging structured economic theory with unstructured data via deep learning**.  
