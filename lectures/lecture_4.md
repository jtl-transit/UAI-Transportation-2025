# Lecture 4: Generative AI for Transportation

Lecture links:

- [**Section 1:** AI to Understand Bus Operators' Preferences](https://youtu.be/ESqUjX_PLh4)
- [**Section 2:** AI for Generative Urban Design](https://youtu.be/YktMW9raePg)
- [**Section 3:** AI for Mobility Trajectory Generation](https://youtu.be/w6yHN9g4RQ0)

# Automatic Transcript

## Section 1: AI to Understand Bus Operators' Preferences

### Introduction  
Welcome back to *Artificial Intelligence for Transportation.*  
Today is Lecture 3, where I’d like to bring you from **AI models and approaches at MIT** to **real-world AI applications.**

At MIT, we’ve had the privilege of working with transportation operators and authorities around the world, often for more than a decade. Examples include:  
- Transport for London (United Kingdom)  
- Chicago Transit Authority  
- Hong Kong MTR  
- Boston MBTA (our home system)  
- Singapore Land Transport Authority  
- Washington, D.C. WMATA  

In these collaborations, some ideas have been welcomed and implemented wholeheartedly, others were initially dismissed as useless, and some faced resistance before being accepted over time.  

This process allows us to engage in what I call **cross-cultural learning**: MIT doesn’t just export scholarship, but also acts as an intermediary—testing ideas in one city, refining them in another, and transferring lessons back to others.  

---

### Research Buckets  
We categorize our research into three main areas:  

1. **Performance-related**  
   - Understanding origin-destination flows  
   - Measuring demand and capacity  

2. **Behavior-related**  
   - Customer segmentation  
   - Responses to disruptions  
   - Effects of crowding on services  

3. **Decision support**  
   - Transit simulation systems  
   - Operations control tools  
   - Crew scheduling and labor models  

---

### Application Levels  
We also work across **three hierarchical levels**:  

1. **Operational level**  
   - Bus dispatching at terminals  
   - Crew scheduling  
   - Service planning  

2. **Tactical level**  
   - Fare policy  
   - Ridership demand management  

3. **Strategic level**  
   - Long-term planning and financing  
   - Integration of new technologies (e.g., autonomous vehicles)  
   - National or regional transportation planning  

---

### Example: Transport for London (TfL)  
Our work with London spans all three levels:  

- **Customer behavior and predictive analytics**  
  - Multimodal origin-destination estimation  
  - Reliability monitoring and public reporting  
  - Predictive demand analytics for the Underground  

- **Decision support and performance measurement**  
  - Smart card–based travel pattern recognition  
  - Disruption management  
  - Bus operations under road construction  

- **Planning and policy**  
  - Transit connectivity and communication capacity  
  - Integrating AVs with public transit  
  - Improving demand-responsive networks  

---

### Case Studies Overview  
Today I’ll present **three case studies**:  
1. Using AI to detect and predict **platform crowding**  
2. Using AI to improve **bus reliability** (avoiding bus bunching)  
3. Using AI to understand **bus operator preferences**  

---

### Case Study 1: Platform Crowding Detection  

#### Why Platform Crowding Matters  
- **Safety:** crowded platforms can endanger passengers and staff.  
- **Operations:** crowding leads to delays and inefficiencies.  
- **Customer experience:** travelers need accurate information to plan journeys.  

#### Hong Kong MTR Example  
- At Admiralty Station (PM peak, Tsuen Wan line), **80–90% of passengers cannot board the first train.**  
- The Hong Kong government requires MTR to report these “left-behind” statistics.  
- Historically, this was done via **manual surveys**—staff literally counting passengers left behind.  

MIT worked with MTR to **automate the process** using:  
- Automatic Fare Collection (AFC) data  
- Automatic Vehicle Location (AVL) data  
- AI-based estimation engines  

Results:  
- Predictions aligned well with manual survey data.  
- Methodology now deployable at scale with near-zero marginal cost.  
- Supports two functions:  
  1. Customer information (expected delays, boarding probability)  
  2. Government reporting  

---

#### Washington, D.C. WMATA Example  
WMATA data sources include:  
- AVL systems  
- Scheduled train movements  
- AFC systems  
- Weather data  
- Planned events (sports, concerts, etc.)  
- Service adjustments  
- CCTV (camera data)  

##### Innovation 1: Operational Data + AI  
- Baseline models (historical averages) aligned with reality in normal times.  
- But during **special events**, predictions failed dramatically.  
- AI (XGBoost) matched ground truth in both normal and high-crowding conditions.  

##### Innovation 2: CCTV Integration  
- Adding real-time CCTV data boosted accuracy by **30% over baseline.**  
- Used **image segmentation** (area covered by passengers) instead of identifying individuals.  
- This method is both low-cost and computationally efficient.  

##### Innovation 3: Conditional CCTV Use  
- CCTV analysis is computationally expensive.  
- New strategy: **only trigger CCTV when prediction uncertainty is high.**  
- Most of the time, rely on traditional data sources.  
- This balances accuracy with resource efficiency.  

---

### Ethical Considerations  
- **De-identification:** No individual-level data; only aggregate measures.  
- **Responsible AI:** Ensure that computer vision respects privacy.  

---

### Lessons from History in Prediction  
Three observations about using history in AI models:  

1. **History repeats itself.**  
   - Historical averages often provide strong baselines.  

2. **Selective history works best.**  
   - Use the most relevant pieces of history for current conditions.  

3. **History is decomposable.**  
   - Full historical events rarely repeat, but components can be recombined to forecast the future.  

---

### Transition to Next Case Study  
With this, we’ve seen how AI can transform **platform crowding prediction**—enhancing safety, operations, and customer experience.  

Next, we’ll move to **Case Study 2: improving bus service reliability and avoiding bus bunching.**

## Section 2: AI for Generative Urban Design

### Introduction  
Our second example explores **AI for generative urban design.**

Consider the **High Line in New York City**—before and after its conversion. The transformation illustrates how urban design influences both people’s appreciation of space and their travel behavior.  

Another contrast: **Atlanta vs. Barcelona.**  
- Both cities have similar populations.  
- But Atlanta is ~30 times larger in land area.  
- Differences in the built environment directly affect behavior, energy consumption, and CO₂ emissions.  

This shows why the **design process is central**—not just for livability, but also for shaping travel behavior and environmental impact.  

---

### The Traditional Planning Process  
Urban planners consider zoning, policy, traffic, social interaction, natural environment, and infrastructure.  
- A plan is drafted, visualized, and presented.  
- Public engagement follows: residents offer suggestions.  
- Planners then absorb feedback, commission new renderings, and return months later with an updated plan.  

**Problems:**  
- Iterations take months.  
- Public momentum and memory fade.  
- Dialogue is slowed and often disengaging.  

**Question:** Can AI **expedite this process** and make it more interactive?  

---

### Example: Public Engagement in Cambridge  
- A planner presents a design.  
- A resident suggests moving Bus Route 25 north by one block for school access.  
- Another resident requests a bike path along the park.  

Traditionally, planners respond: *“Good ideas, but give me three months to revise the plan.”*  
By then, momentum is lost.  

With AI, however, planners could **regenerate designs on the fly** based on feedback—making dialogue immediate, engaging, and meaningful.  

---

### What Planners Want  
Planners generally require three things from an AI tool:  

1. **Control** over generated land-use proportions.  
   - e.g., 35% residential, 15% commercial, 10% parks.  

2. **Respect for existing infrastructure and natural environment.**  
   - Rivers, railways, and roads must be preserved.  

3. **Ability to learn and apply different urban textures.**  
   - Chicago style vs. Dallas style vs. Los Angeles style.  

---

### Technical Approach  
We developed a **Stable Diffusion + ControlNet tool**, using **OpenStreetMap data** as ground truth.  
- OpenStreetMap provides open-source land-use, infrastructure, and environmental data.  
- Ensures generalizability and reproducibility.  

The AI tool can:  
- Generate designs respecting user-specified land use proportions.  
- Integrate with existing infrastructure (e.g., waterways, roads).  
- Reproduce different urban textures/styles.  

---

### Demonstrations  

#### Criterion 1: Control Over Land Use  
- Example: 35% residential, 15% commercial, 10% parks.  
- AI generates corresponding designs instantly.  
- Adjustments (e.g., 65% industrial, 20% commercial, 5% parks) produce new layouts.  
- Traditional manual methods would take months.  

#### Criterion 2: Respecting Infrastructure  
- Example: Downtown Chicago.  
- Design must preserve the **Chicago River**, railways, and roads.  
- AI generates layouts adhering to both land use proportions and existing constraints.  
- Comparison with satellite images shows strong similarity.  

#### Criterion 3: Applying Urban Textures  
- AI generates Chicago-style layouts.  
- Can switch to Dallas or Los Angeles styles while still respecting constraints.  
- Captures urban “essence” and adapts it to new contexts.  

**Bonus:** AI generates multiple design alternatives instantly (Design 1, Design 2, Design 3).  

---

### Implications for Planners  

1. **Immediacy changes dialogue.**  
   - No more *“come back in three months.”*  
   - Planners can respond in real time to suggestions.  

2. **Public empowerment.**  
   - Previously, only planners had technical control.  
   - With AI, residents can generate their own alternatives during engagement.  
   - Shifts power balance, enabling more meaningful participation.  

3. **Changing role of planners.**  
   - From **creator → critic**: evaluating designs rather than solely producing them.  
   - From **creator → communicator**: explaining rationale to the public and decision makers.  
   - From **creator → negotiator**: facilitating stakeholder discussions, since planning is fundamentally about power allocation.  

---

### Human + AI Collaboration  

Urban design is an **iterative process**:  
- Start with design objectives and site constraints.  
- Generate drafts.  
- Propose schemes.  
- Conduct public engagement.  
- Adjust objectives.  
- Iterate.  

**Goal:** Embed human input meaningfully throughout.  

---

### Human-Centered AI for Urban Design  
We propose a **step-wise, human-in-the-loop framework**:  

1. **Stage 1: Road Network & Land Use**  
   - Human provides prompts (e.g., 70% residential, 20% commercial, 10% parks).  
   - AI generates ~10 alternatives.  
   - Human selects the most suitable one.  

2. **Stage 2: Building Layout Planning**  
   - Based on selected option, human prompts AI to add building layouts.  
   - AI generates new options.  
   - Human evaluates and selects.  

3. **Stage 3: Detailed Planning & Rendering**  
   - AI generates detailed final layouts and visualizations.  
   - Human again chooses among options.  

---

### Conclusion  
- AI enables **fast, responsive, and participatory design.**  
- Human-in-the-loop ensures diversity, control, and contextual sensitivity.  
- Benefits:  
  - Efficiency in urban design  
  - Enhanced public communication  
  - Iterative refinement  
  - More balanced allocation of power in planning  

In short, human-centered AI for urban design combines **the efficiency of AI** with **the judgment and negotiation skills of planners**, leading to more livable, inclusive, and adaptive cities.

## Section 3: AI for Mobility Trajectory Generation

### Introduction  
Our third example explores how to use **AI for mobility trajectory generation.**

Urban mobility data are essential to understand individual travel behavior and are strongly linked to urban economic growth. But generating large-scale trajectory datasets is:  
- **Expensive** to produce  
- **Privacy-sensitive**, as they often involve personal movement data  

**Solution:** Generate **synthetic data** instead of relying solely on real data.  

The goal:  
- The synthetic data should match the **distribution** of the real data.  
- Even though individual records differ, the overall patterns should be the same.  

These synthetic trajectories can then be used to:  
- Fine-tune large language models (using synthetic text data)  
- Train autonomous driving algorithms (using synthetic vision or trajectory data)  

---

### Challenge in Trajectory Generation  
Mobility trajectories are difficult to synthesize because:  
- Human activities are diverse and complex.  
- The generated data must satisfy **road network constraints**.  

Our approach:  
- Leverage a **hierarchical trajectory structure** to reduce computational complexity.  
- Move from coarse-grained road segments → fine-grained GPS sequences.  

---

### Analogy: Multi-Resolution Image Generation  
- In image generation, models move from **low resolution (16×16)** → **medium (256×256)** → **high resolution (1024×1024)**.  
- Similarly, in mobility:  
  - **Coarse-grained:** road segments (link by link)  
  - **Fine-grained:** GPS sequences along each road link  

This hierarchy simplifies computation and ensures physical realism.  

---

### Model: Cascaded Hybrid Diffusion  
We developed a **cascaded hybrid diffusion model** that generates trajectories in two stages:  

1. **Segment level (coarse)**  
   - Use **latent diffusion** with a latent transformer.  
   - Generates discrete road segment sequences while respecting road network constraints.  

2. **GPS level (fine)**  
   - Use **continuous diffusion** for detailed GPS trajectories along each road segment.  
   - Ensures adherence to road structure and conditional information (e.g., travel time).  

The denoising process works in stages:  
- At the coarse level: start from random noise → gradually form the road structure.  
- At the fine level: refine into continuous GPS trajectories conditioned on the coarse structure.  

---

### Results  
- Our cascaded diffusion model outperformed three alternative models.  
- Synthetic trajectories matched real data distributions much more closely.  
- Applied successfully to datasets from **Singapore, Chengdu, and Porto**.  
- Enabled **conditional generation**:  
  - Hard constraints: specific origins and destinations  
  - Soft constraints: travel time ranges  
  - Result: realistic synthetic trajectories aligned with both spatial and behavioral constraints  

---

### Broader Applications of Generative AI in Transportation  
We’ve now seen three examples:  
1. AI to analyze **preferences via language**  
2. AI to generate **urban designs**  
3. AI to generate **mobility trajectories**  

But generative AI has far broader potential across transportation, particularly in **intelligent transportation systems (ITS):**  

#### Four Core Areas  
1. **Traffic Perception**  
   - Imputing missing data  
   - Estimating traffic conditions  
   - Anomaly detection  

2. **Traffic Prediction**  
   - Individual human behavior prediction  
   - Collective demand forecasting  
   - Vehicle trajectory prediction (as shown in Example 3)  
   - Road segment-level predictions  

3. **Traffic Simulation**  
   - Simulating driver behavior  
   - Scenario testing  
   - Traffic flow generation  

4. **Traffic Decision-Making**  
   - Road planning  
   - Traffic signal control  
   - Vehicle-level control  

---

### Full Chain of Transportation Functions  
AI can support nine interlinked functions:  
1. Data fusion and analysis  
2. Travel demand management  
3. Traffic prediction  
4. Simulation  
5. Decision-making  
6. Environmental modeling  
7. Safety and risk management  
8. Public engagement  
9. Long-term planning  

Across all these, AI provides:  
- **Scalability**: Automating tasks that are difficult for humans to scale  
- **Adaptability**: Adjusting to changing conditions  
- **Efficiency**: Handling massive datasets and scenarios quickly  

---

### Generative AI Modalities in Transportation  
- **Text generation**  
  - Writing traffic reports  
  - Interpreting incident alerts  

- **Image generation**  
  - Creating traffic scenarios with varied weather and lighting conditions  

- **Video generation**  
  - Generating synthetic traffic videos  
  - Testing autonomous vehicle responses in diverse environments  

---

### Opportunities and Concerns  

#### Opportunities  
- Handle vast scenarios at scale  
- Automate and accelerate processes  
- Personalize services and resources  
- Enable creative design of future mobility systems  

#### Concerns  
- **Resource requirements**: Training and compute costs are significant  
- **Workforce impact**: Potential for job displacement  
- **Equity risks**: Personalization may marginalize certain groups  
- **Transparency issues**: Black-box AI raises interpretability concerns  
- **Creativity risks**: Over-reliance on AI may stifle human imagination  

---

### Conclusion  
Generative AI is rapidly evolving, and its applications in transportation are still in their **infancy.**  

- The potential is vast: from traffic perception and simulation to decision-making and design.  
- But challenges remain in scalability, transparency, equity, and human creativity.  

Thank you for listening. This concludes the lecture on **Generative AI for Transportation.**  
See you in the next lecture.  
