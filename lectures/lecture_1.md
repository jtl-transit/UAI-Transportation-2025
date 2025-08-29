# Lecture 1: Introduction

Lecture links:

- [**Section 1:** Introduction](https://youtu.be/XkjXAa03G98)
- [**Section 2:** AI Models and Approaches](https://youtu.be/1tPa6VCPuEM)
- [**Section 3:** Real-World AI Case Studies](https://youtu.be/Z-6LcpXjgfA)
- [**Section 4:** Generative AI](https://youtu.be/6NQIihXJZ1Y)

# Automatic Transcript

## Section 1: Introduction

### Transcript: Introduction to Artificial Intelligence for Transportation  

Hello everyone, and welcome to the class *Artificial Intelligence for Transportation.*  

Before I begin, I’d like to ask you two questions:  
1. What defines success in transportation?  
2. What defines the future of transportation?  

If you look at this diagram, you’ll see that throughout human history, we’ve invented a fantastic set of technologies to help us move from point A to point B. But here’s the critical question: have these technological advances actually translated into a better transportation system?  

If you live in Boston, the answer is clearly no. Almost everybody in Boston complains about congestion and the problems with the MBTA. People are not happy with the transportation system. And this isn’t unique to Boston.  

Take Beijing, for example. Here’s a picture of Beijing today: the city is dominated by cars, with congestion, pollution, traffic accidents, and all the familiar issues. Now compare that with a picture of the same city 40 years ago. What do you see? Bicycles, pedestrians, public transit—and by the way, they were all electrified.  

How do we describe this picture in transportation planning jargon? We’d call it *sustainable travel, low-carbon travel, healthy travel, green travel.* All the positive terms we like to associate with good transportation. This older picture actually represents the paradigm that modern transportation planners aspire to achieve.  

Yet Beijing, as a city, has deliberately moved away from this sustainable paradigm, toward the car-dominated model we see across much of the world today. So, as a profession, I would argue we’ve confused what “success” really means.  


### Three Questions  

**Question 1:** How many people were killed in road traffic crashes in the United States in 2021?  
The number is **42,915.**  

Is this number good or bad? Let’s put it in context. In the same year, in the U.S. commercial airline industry, how many people died? **Zero.**  

That’s puzzling. Both transportation systems, both about safety, but such a stark contrast. This isn’t just an academic curiosity—it’s a practical issue. With new technologies like electric vehicles, autonomous vehicles, and roadside robotics, we need to ask: *where will these technologies place us along the spectrum between zero and 42,000 deaths?*  

**Question 2:** How much does transportation contribute to greenhouse gas emissions in the U.S.?  
As of 2022, the U.S. generated **6.3 billion tons of CO₂,** of which **1.8 billion tons came from transportation.** That means transportation accounts for **28% of U.S. emissions**—far higher than the global average of 16%. And that’s largely due to our car-dominated transportation culture.  

**Question 3:** Is public transit affordable in Boston?  
Many people think of Boston as a wealthy city, but in fact, many residents cannot afford bus fare.  


### Taking Stock  

So, are we truly successful? Let’s recap:  
- Cities are congested.  
- 42,000 people die in crashes every year.  
- 28% of greenhouse gas emissions come from transportation.  
- Many people can’t afford basic transit.  

Despite tremendous technological progress, these advances haven’t translated into a better transportation system.  


### Looking Ahead: What Defines the Future?  

My answer is simple: **behavior** and **computation.**  

- **Behavioral thinking**: asking questions like, Is travel social? Is it emotional? Is time absolute? We need a deeper understanding of how people actually make decisions in the transportation domain. Why do people travel the way they do? What are the consequences? And is there a way we can help improve these decisions?  

- **Computational thinking**: applying data, algorithms, and computing. AI is booming, and it offers five fundamental functions:  
  1. Represent  
  2. Predict  
  3. Explain  
  4. Control  
  5. Create  

The future lies in bringing these two dimensions together:  
- Behavioral thinking—understanding travel as social, emotional, perceptual.  
- Transportation technologies—electrification, automation, connectivity, sharing.  
- Computational foundations—data, algorithms, and AI.  


### Course Overview  

This module on *AI for Transportation* will have four lectures:  
1. Today’s introduction.  
2. Next, AI models and approaches.  
3. Then, real-world examples of AI applications.  
4. Finally, generative AI.  

In today’s lecture, we’ve just covered the first section: *what defines success, and what defines the future.*  

Next, I’ll discuss what a transportation system is, what changes are happening, and what data we rely on. Then, I’ll move on to grounded AI applications, and lastly, we’ll examine some of the tensions and opportunities we see in applying AI to transportation.  

## Section 2: AI Models and Approaches

### What is Transportation?  
Transportation is the **movement of people and goods.**  
It covers a wide range of activities:  
- Urban mobility  
- Intercity travel  
- The airline industry  
- Ocean shipping  

### What is a Transportation System?  
A transportation system is more than just roads and vehicles. It includes:  

1. **Physical framework** – the part most people imagine when they think of transportation:  
   - Vehicles  
   - Infrastructure  
   - Energy systems  

2. **Functional frameworks** – the less visible but critical structures:  
   - Technology framework  
   - Behavior framework  
   - Political framework  
   - Financial framework  
   - Environmental framework  

3. **Stakeholder system** – often overlooked but essential:  
   - Businesses  
   - Government sectors  
   - Civil society  

All these elements together enable the delivery of mobility for people.  


### What’s Changing in Transportation?  
The changes can be captured in three words: **technology, data, and value.**  

- **Technology:** automation, electrification, connectivity, and sharing.  
- **Data:**  
  - Ubiquitous sensing  
  - AI and computing applied to transport data  
  - Cybersecurity challenges  
  - Questions about data ownership and property rights  
- **Value:** transportation is no longer just a congestion problem.  
  It’s also:  
  - A climate change problem  
  - A future of work problem  
  - A public health problem  
  - A social justice problem  
  - An urban livability problem  

This means transportation must now be understood in **multi-dimensional terms**—not just congestion and efficiency, but also sustainability, justice, identity, creativity, and public health.  


### A Case Study: Transport for London (TfL)  
London offers a good example of how **data and analytics** have evolved in transportation.  

### Data Sources in Transportation  
- Traditional traffic surveys  
- Smart card data (e.g., Oyster card)  
- Automatic vehicle location data  
- Mobile phone data  
- Incident logs, customer feedback  
- Social media streams (Twitter, Facebook, YouTube, etc.)  

This data captures:  
- **Vehicles** (location, movement, condition)  
- **Travelers** (surveys, behavioral data, preferences)  
- **Sensors** (loop detectors, cameras, microphones, mobile sensors)  

### Evolution of Data at TfL  
- **Phase 1:** Manual surveys as the primary data source.  
  - Slow, retrospective (by the time data was processed, it was already outdated).  
- **Phase 2:** Smart card–centric data platform.  
  - MIT worked with TfL beginning in 2006 to make this transition.  
  - Oyster card became the backbone of London’s real-time mobility tracking.  
- **Phase 3:** Multimodal data fusion.  
  - Combining traffic, transit, survey, and digital data sources.  
  - Moving toward predictive analytics.  
  - Goal: take action before problems (e.g., congestion, delays) occur.  

With this richness of data, TfL is moving beyond retrospective analysis toward **real-time and predictive analytics.**


### A Multi-Channel View of Cities  
Most of us know Plato’s *Allegory of the Cave*—that what we see is often only a shadow of reality. Cities can be understood in a similar way.  

- Some people look at a city through **numbers** (population size, car ownership, miles of road).  
- Others view it as a **graph** (networks of roads and transit).  
- Designers often see it through **images** (maps, aerial views).  
- Writers may describe it through **language** (stories, books, cultural narratives).  

Different professions use different lenses, and each perspective captures only part of the city.  

### Opportunity with AI  
Traditionally, prediction models in transportation look like this:  

   \[ 
   P(Y \mid X)  
   \] 

where **Y** is the outcome (e.g., congestion, delay, car ownership) and **X** is the data—usually *numerical variables* (income, education, number of cars, miles of road).  

But cities are richer than just numbers. They are also described by:  
- **Images**  
- **Language**  
- **Graphs**  

With modern AI—deep neural networks and large-scale models—we now have the ability to **unify these channels.** That means treating numbers, images, language, and graphs all as independent variables to build a deeper, more comprehensive understanding of cities.  

## Section 3: Real-World AI Case Studies

### Structuring the Discussion  
In this section, we’ll cover:  
1. The wide range of **transportation topics** where AI is applied.  
2. The **leading trends** in the mobility industry.  
3. The question of **“why now”**—why we are discussing AI applications in transportation today.  
4. Finally, what we mean by **grounded AI.**  


### Transportation Topics & AI Applications  
Transportation touches many areas: congestion pricing, autonomous vehicles, electric vehicles, urban rail, fleet management, and more.  

AI is already being applied in domains such as:  
- Traffic management  
- On-demand mobility operations  
- Automated and assisted driving  
- Predictive infrastructure maintenance  
- EV charging management  
- Smart parking  

AI is not new to transportation. Since the 1980s, we’ve discussed “intelligent transportation systems.” But today, AI offers fundamentally new capabilities, which is why we’re refreshing our thinking in this class.  


### The Progression of AI  
Jensen Huang (NVIDIA) describes AI development as a progression:  
- **Perception AI** (early computer vision, speech)  
- **Generative AI**  
- **Agentic AI**  
- **Physically embodied AI** (e.g., autonomous vehicles)  

When **data, algorithms, and computing** come together, AI becomes powerful. Remember: AI has five core functions:  
1. Represent  
2. Predict  
3. Explain  
4. Control  
5. Create  

The key question is: **how do these functions interact with the needs of the transportation system?**  


### A Framework for Transportation Analytics  
At the center, we have **transportation analytics**: monitoring, evaluation, prediction, and optimization.  

#### Long-term (offline) functions:  
- **Supply side:** service and operations planning (e.g., how many roads to build, how many trains to run).  
- **Demand side:** influencing land use, employment distribution, and urban activities.  

#### Real-time functions:  
- **Supply side:** operational control—adjusting services on the fly.  
- **Demand side:** providing real-time information to influence passenger behavior.  

Both supply and demand generate data:  
- **Supply:** tracked by tools like automatic vehicle location (AVL) systems.  
- **Demand:** captured through fare collection systems (e.g., smart cards) or passenger counting systems.  

All of this feeds into transportation analytics, completing the cycle of monitoring, prediction, and optimization.  


### Exercise for Students  
Each student should **create a similar framework** for their area of interest—be it public transit, autonomous vehicles, electric vehicles, or urban drones. Use the supply/demand and offline/real-time structure to capture the system dynamics, and then consider how AI functions (represent, predict, explain, control, create) map onto them.  


## AI Functions × Human-AI Personas  
If we add another dimension—**human-AI interaction**—we get a two-dimensional framework:  

- **AI functions:** represent, predict, explain, control, create  
- **Human-AI personas:** sense, assist, work, instruct, interface  

Each cell in this grid represents an **AI application opportunity.**  


### Examples of Grounded AI Applications  
1. **Sentiment analysis (Washington, D.C.)**  
   - With WMATA (the D.C. transit authority), we analyzed how riders express opinions on Twitter and other platforms.  
   - AI’s *represent* function helps interpret customer sentiment.  

2. **Crowd detection with computer vision (major stations)**  
   - AI predicts and senses crowding, assisting operational decisions.  

3. **Reinforcement learning for bus dispatching (Chicago)**  
   - Applied RL to reduce bus bunching and improve service reliability.  

4. **Generative AI for future cities**  
   - Using generative capabilities to imagine alternative city futures.  

5. **Workforce planning with AI**  
   - Understanding employee preferences and dynamics in transit agencies.  


### Key Trends in the Mobility Industry  
AI enables several major shifts:  
1. From **retrospective** to **predictive** analysis.  
2. From **mass services** to **individualized services.**  
3. From **static planning** to **dynamic, experimental planning.**  
4. From **isolated modal views** to **integrated multimodal network analysis.**  

All of these are driven by **behavioral understanding** and **computational capacity.**  


### Why Now?  
We are talking about AI in transportation today because of four converging factors:  

1. **Technology maturity** – AI has advanced to the point of industry-wide deployment.  
2. **Market readiness** – Investors and companies are putting enormous resources behind AI.  
3. **Societal pressure** – Climate change, congestion, safety, and inequality create urgent needs.  
4. **Policy flux** – Governments are struggling to regulate AI, but the frameworks are emerging.  


### What is Grounded AI?  
When I say **grounded AI**, I mean AI that is:  

- **Problem-driven, not technology-driven** – addressing real-world needs.  
- **Context-aware** – embedding institutional, human, and social considerations into models.  
- **Human-in-the-loop** – keeping human oversight and agency central.  
- **Deployment-ready** – designed for real-world use by industry and communities.  
- **Iterative** – co-created with developers, industry partners, and stakeholders.  


### Questions for Students on Grounded AI  
If you are interested in developing grounded AI applications, ask yourself:  
1. After you develop a model, **can you explain it to a layperson?**  
2. When the model works, **who benefits?** When it fails, **who loses?**  
3. Once AI is deployed, **who remains in control?**  

By reflecting on these questions, we can better align AI with the real-world principles and responsibilities of transportation systems.  

## Section 4: Generative AI

### Tensions in AI Modeling  
There are two main schools of thought in model development:  

1. **Domain-specific models**  
   - Built on decades of human knowledge, theory, and carefully chosen data.  
   - Example: transportation models grounded in economic and behavioral theory.  

2. **Learning-based models**  
   - Rely less on theory and more on large volumes of data.  
   - Patterns are discovered automatically by algorithms (e.g., deep neural networks).  

Traditionally, the conventional wisdom has been:  
- **Machine learning (ML) models** excel at **prediction.**  
- **Domain-specific models** excel at **interpretation and robustness.**  

But these distinctions are breaking down. Advances in both paradigms are challenging the old assumptions, and increasingly, each side is borrowing from the other.  


### Example: Travel Behavior Analysis  
When predicting travel behavior, two types of errors matter:  

- **Approximation error:** can the model represent the richness of human travel behavior?  
- **Estimation error:** can the model parameters be reliably learned from empirical data?  

These two errors often trade off against each other.  
- More complex models → lower approximation error (better representation of reality), but higher estimation error (harder to estimate reliably).  
- Simpler models → higher approximation error, but lower estimation error.  

### Comparing Models  
- **Multinomial Logit (MNL) model:**  
  - Simple, elegant, interpretable.  
  - High approximation error, low estimation error.  
  - Example: predicting whether someone will drive, take a bus, walk, bike, or use Uber from Lexington to Cambridge. As driving cost increases, probability of driving declines, while transit and walking rise—exactly as intuition suggests.  

- **Deep Neural Network (DNN):**  
  - Very high capacity (can capture infinite complexity).  
  - Low approximation error but high estimation error.  
  - Produces rich but messy curves that are often difficult to interpret.  

This illustrates the **tension**: structured simplicity versus flexible complexity.  


### Toward Synergy  
A promising approach is to combine the two. For example:  
- **Residual Neural Networks (ResNets):**  
  - Integrate domain-specific structures with the flexibility of DNNs.  
  - Capture broad behavioral patterns while also incorporating local nuances.  
  - Achieve higher prediction accuracy and improved representation.  

This hybrid approach leverages the strengths of both paradigms.  


### Criteria for a Good Model  
1. **Prediction accuracy** – must be competitive.  
2. **Interpretability** – humans need to understand the model.  
   - One aspect is **sparsity:** reducing millions of variables to a handful (ideally 5–7) so humans can reason about them.  
3. **Robustness** – models should not be fragile to noise or local disturbances.  
4. **Practicality** – models developed in research must be usable in practice (e.g., MBTA should be able to adopt an MIT model without difficulty).  


### Interpretability and Its Roles  
Models serve two roles:  

1. **Substantive role:** helping us answer “how many autonomous vehicles are needed in Arlington, MA?”  
2. **Communicative role:** explaining results to mayors, transit directors, residents, and passengers.  

If a model predicts well but cannot communicate clearly, it fails the second role.  


### Methods to Improve Interpretability  

#### Intrinsic methods (built into model design):  
- Monotonic constraints  
- Causal inference frameworks  
- Uncertainty quantification  

#### Post-hoc methods (after training):  
- Feature attribution (e.g., SHAP values)  
- Local linear approximations  
- Saliency maps (highlight where the AI is “looking”)  
- Counterfactual explanations  
- Concept-based methods  

#### Example: Sparsity  
- Predicting next-hour Uber demand at a location.  
- Without constraints, the model might use 168 prior hours as predictors—far too many for humans to grasp.  
- By enforcing sparsity (say, limiting to 5 factors), the model identifies the most influential drivers:  
  - The same hour one week ago (weekly periodicity).  
  - The same hour one day ago (daily cycle).  
- This preserves interpretability while keeping predictive power.  

#### Example: Orthogonality  
- Decomposing models into independent factors (via matrix or tensor factorization).  
- Easier to explain what each factor contributes.  


### Broader Questions in Modeling  
- Should models aim to **predict** or **explain**?  
- Are models primarily **correlational** or **causal**?  
- Should we move from **simple to complex** models, or simplify complex models?  
- Are models for **understanding** or for **action**?  
- Is the purpose of modeling to **discover** or to **create** knowledge?  


### Opportunities in AI for Transportation  
Despite tensions, opportunities lie in integrating behavior, domain knowledge, and computation.  

#### AI contributes in three main buckets:  
1. **Core capabilities** – representing, predicting, explaining, controlling, and creating.  
   - AI is powerful and should be embraced.  
2. **Communication** – the more complex the model, the harder it is to explain to decision makers and the public.  
   - This is where caution is needed.  
3. **Defining success** – ultimately, humans must remain in control of objectives.  
   - AI should assist, not dictate.  


### Conclusion  
In this lecture we:  
- Defined what success and the future of transportation mean.  
- Examined what constitutes a transportation system.  
- Discussed grounded AI applications.  
- Explored tensions and opportunities in AI.  

Moving forward, this course will cover:  
1. AI models and approaches.  
2. Real-world AI case studies.  
3. Generative AI. 
