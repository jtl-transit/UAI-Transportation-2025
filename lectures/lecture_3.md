# Lecture 3: Case Studies

Lecture links:

- [**Section 1:** AI to Detect Platform Crowding](https://youtu.be/ESqUjX_PLh4) 
- [**Section 2:** AI for Bus & Train Dispatching Control](https://youtu.be/DlfkRDiM_Ag)
- [**Section 3:** AI to Understand Bus Operators' Preferences](https://youtu.be/nuhq70FWPRM)

# Automatic Transcript

## Section 1: AI to Detect Platform Crowding

### Introduction  
Welcome back to *Artificial Intelligence for Transportation.*  
Today is Lecture 3.  

In this lecture, I will move from the AI models and approaches we discussed at MIT to **real-world AI applications.**  

At MIT, we have the privilege of working with transportation operators and authorities worldwide—often in partnerships lasting over a decade. Examples include:  
- Transport for London (TfL), United Kingdom  
- Chicago Transit Authority  
- Hong Kong MTR  
- Boston MBTA (our home system)  
- Singapore Land Transport Authority  
- Washington, D.C. WMATA  

Through these collaborations, we’ve seen different reactions: some ideas were welcomed and implemented immediately, others were initially dismissed as useless, and some faced resistance before eventually gaining acceptance.  

This global experience allowed us to facilitate what I call **cross-cultural learning.** MIT doesn’t just produce scholarship—it also acts as a conduit, bringing ideas from one city to another, refining them, and enabling shared progress.  

---

### Research Categories  
We categorize our research into three major areas:  

1. **Performance-related**  
   - Origin-destination analysis  
   - Demand and capacity understanding  

2. **Behavior-related**  
   - Customer segmentation  
   - Responses during disruptions  
   - Effects of crowding on behavior  

3. **Decision support**  
   - Simulation systems  
   - Operations control tools  
   - Crew scheduling and labor systems  

---

### Application Levels  
We also think about applications across three levels:  

1. **Operational level**  
   - Bus dispatching from terminals  
   - Crew scheduling  
   - Service planning  

2. **Tactical level**  
   - Fare policy and ridership demand  
   - Travel demand management  

3. **Strategic level**  
   - Long-term planning and financing  
   - Regional or national policy  
   - Integration of new technologies like AVs into transit systems  

Our research and applications bridge all three levels—operational, tactical, and strategic.  

---

### Examples: Transport for London  
Working with TfL, we’ve contributed at multiple levels:  

- **Customer behavior and predictive analytics**  
  - Multimodal origin-destination estimation  
  - Reliability monitoring and reporting  
  - Predictive analytics for Underground demand  

- **Decision support and performance measurement**  
  - Smart card–based travel pattern recognition  
  - Disruption management  
  - Bus operations under road construction  

- **Planning and policy**  
  - Transit connectivity and communication capacity  
  - Integration of AVs with public transit  
  - Demand-responsive transit improvements  

---

### Case Studies (Three Examples)  
Today I’ll share three real-world examples of AI applications:  

1. **Platform crowding detection and prediction**  
2. **Service reliability and bus bunching prevention**  
3. **Understanding bus operator preferences**  

---

### Case Study 1: Platform Crowding Detection  

#### The Problem  
In many transit systems—Hong Kong, London, Chicago—stations frequently operate near capacity. Crowding creates three challenges:  
1. **Safety** – high-density platforms can endanger passengers and staff.  
2. **Operational efficiency** – delays ripple through the system.  
3. **Customer experience** – passengers need accurate information for journey planning.  

#### Example: Hong Kong MTR  
- At Admiralty Station during the PM peak, **80–90% of passengers cannot board the first train.**  
- By law, MTR must report these statistics to the government.  
- Traditionally, this was done through **manual surveys**—staff counting passengers left behind.  

MIT worked with MTR to automate this process using:  
- Automatic fare collection (AFC) data  
- Automatic vehicle location (AVL) data  
- AI-driven estimation engines  

Results:  
- Automated predictions aligned well with manual surveys.  
- Enabled **real-time customer information systems** and **government reporting** at scale and near-zero marginal cost.  

---

#### Example: Washington, D.C. WMATA  
Data sources used:  
- AVL system  
- Scheduled railway movements  
- AFC data  
- Weather data  
- Event schedules (sports, concerts, etc.)  
- Revenue service adjustments  
- CCTV footage  

##### Innovations Introduced  
1. **AI-enhanced prediction**  
   - Compared to baseline predictions (historical averages), AI (using XGBoost) captured unusual surges during special events.  
   - Critical because baseline methods failed precisely when predictions mattered most.  

2. **Incorporating CCTV data**  
   - Computer vision segmented crowd areas to estimate density.  
   - Improved prediction accuracy by over **30%** compared to baseline.  

3. **Selective use of CCTV**  
   - CCTV analysis is computationally expensive.  
   - AI was trained to trigger CCTV use only when prediction uncertainty was high.  
   - Balanced accuracy with resource efficiency.  

---

### Observations  

#### Ethical Use of AI  
- All camera data was **de-identified.**  
- No individuals were tracked—only aggregate crowding measures were used.  

#### The Role of History  
Prediction relies on historical patterns. We learned three lessons:  
1. **History repeats itself.** Historical averages often work well as baselines.  
2. **History can be selective.** Only use the most relevant historical data for current conditions.  
3. **History is decomposable.** Even if exact events don’t repeat, components of history can be recombined to predict the future.  

---

### Closing Thought  
By integrating operational data, computer vision, and selective AI-driven prediction, we can not only detect and predict crowding but also deliver safer, more efficient, and more customer-friendly transit systems.  

This was just the first example. Next, we’ll look at how AI can improve **bus service reliability** by addressing the bus bunching problem.  

## Section 2: AI for Bus & Train Dispatching Control

### Transcript: AI for Bus Dispatching and Service Reliability (Lecture 3, Example 2)

#### The Bus Bunching Problem  
Let’s start with a problem we all know well: **bus bunching.**  

Take Bus Route #1 on Massachusetts Avenue, right in front of MIT. Many of our students, staff, and faculty rely on this bus, and we’ve all experienced the frustration:  
- You wait… the bus doesn’t come.  
- Then suddenly, three buses arrive at once.  

This is bus bunching.  
Buses are scheduled to come every 15 minutes, but disturbances—a red light, a wheelchair boarding—delay one bus. That bus then picks up more passengers, increasing dwell time, which reinforces delays. The problem cascades.  

Bus bunching happens **everywhere, all the time.**  

---

#### Control Strategy  
The solution: **dispatch control at the terminal.**  
- Hold a bus for 1–3 minutes.  
- Or push forward departure by 1–3 minutes.  
- Goal: even out headways between buses.  

Traditionally, this can be done using heuristics. But we explored a more advanced method: **reinforcement learning (RL).**  

---

#### The Pandemic and Driver Shortages  
During COVID, service reliability got even worse due to:  
- **Driver shortages**: agencies couldn’t hire enough drivers.  
- **Driver absenteeism**: some drivers called in sick the night before, others didn’t show up at all.  

At the Chicago Transit Authority (CTA), only **84% of promised service** was delivered because of absenteeism.  

This made the bunching problem worse—and made the case for AI-driven solutions even stronger.  

---

#### Reinforcement Learning Framework  
- **Agent:** the control center.  
- **Actions:** hold or push forward bus departures.  
- **Environment:** bus locations, passenger loads, waiting passengers.  
- **Reward:** minimize average waiting time or travel time.  

The RL system learns which actions improve service reliability under uncertainty (demand, supply, traffic, compliance).  

---

#### Human in the Loop (Phase 1: Route Selection)  
When we first proposed this project with CTA, we identified **Bus Route 20**—high demand, unreliable service.  

CTA responded: *“Don’t work on Route 20. However clever your AI is, the drivers there won’t listen to you. They follow their own schedules.”*  

Instead, CTA recommended **Route 81**—similar demand and reliability issues, but with more open-minded drivers.  

This was a critical **lesson in human-in-the-loop AI.** AI systems must respect local knowledge and culture.  

---

#### Human in the Loop (Phase 2: Real-Time Recommendations)  
We developed a real-time recommendation system:  
- AI recommends an action (hold or push forward).  
- A **supervisor** at the terminal reviews it.  
- The supervisor then communicates the instruction to the driver.  

Key point: this is **not automated dispatching.**  
- AI recommends.  
- Supervisors and drivers **decide and execute.**  

---

#### Human in the Loop (Phase 3: Field Experiment)  
We piloted this on **Route 81** (north Chicago, 7,500 weekday riders, 6–9 minute headways, 15–25% missed trips due to driver shortage).  

MIT researchers embedded within CTA worked closely with:  
- Control centers  
- Terminal supervisors  
- Garage staff  
- Bus drivers  

With support from leadership to frontline staff, the pilot was possible.  

---

#### Results: Service Reliability  

**Metric 1: Excess Waiting Time**  
- Baseline: passengers often waited much longer than scheduled.  
- With RL control: **37% reduction in excess waiting time.**  
- Equivalent to adding two additional bus trips—without adding buses or drivers.  

**Metric 2: Crowding**  
- Crowding reduced by **5–20%** across bus stops.  
- Evening out headways led to more even passenger distribution.  

**Metric 3: Cycle Time**  
- 90th percentile cycle time reduced by **2.5 minutes.**  
- Improved efficiency, less wasted resource.  

---

#### Compliance: Do Drivers Listen?  
The big question: **do drivers follow AI recommendations?**  

Results showed:  
- Average compliance: **40–50%.**  
- Compliance decreases as the intervention becomes more extreme.  
  - Asking for a 1–3 minute delay: often accepted.  
  - Asking for a 5-minute push forward: almost never accepted.  
- Drivers prefer delays over early departures (they need time for breaks, etc.).  

**Lesson:** assuming 100% compliance is unrealistic. Driver behavior must be built into the model.  

---

#### Phase 2 Research: Embedding Compliance into AI  
- Next step: embed **driver response patterns** into the algorithm.  
- Example: if drivers ignore a 5-minute early departure request, the model should not make such a recommendation.  

We are moving from pilot (MIT-led) to **deployment (CTA staff-led).**  
- MIT trained local staff to operate the system.  
- Scaling from one route to multiple routes.  
- Expanding time periods of operation.  

---

#### Scaling Beyond Chicago  
MIT now coordinates a **Transit Research Consortium** with seven major U.S. transit agencies.  
- Sharing lessons from CTA.  
- Helping other cities adopt this methodology.  

---

#### Human in the Loop: Three Levels  
1. **Control level** – drivers receive recommendations but make final decisions.  
2. **Garage level** – culture matters; some garages are more open to AI guidance than others.  
3. **Organizational level** – broad support (executives to drivers) is essential for success.  

---

#### Conclusion on Bus Reliability AI  
- AI can **substantively improve reliability** without adding buses or drivers.  
- Human compliance is variable and must be modeled.  
- Success depends on embedding **human knowledge and judgment** into AI design and deployment.  

## Section 3: AI to Understand Bus Operators' Preferences

### Transcript: AI for Sentiment Analysis in Transportation (Lecture 3, Example 3)

#### Introduction  
This example looks at how AI can be used for **sentiment analysis** in transportation operations.  

#### Case: Washington, D.C. WMATA  
WMATA (Washington Metropolitan Area Transit Authority) often receives social media messages like this one on Twitter:  

> *“I’m counting over 25 people waiting for a bus at stop 3968. You need to do something to improve this route during weekday afternoon rush hour. It’s insane to spend nearly half an hour just to take a bus.”*  

This is valuable information, but WMATA receives **tens of thousands** of such messages. How can agencies systematically process them?  

---

#### From Narrative to Structured Data  
We developed AI methods to **decode unstructured text into structured data.**  
For example, the tweet above can be classified as:  
- Mode: Bus  
- Route: S2/S9  
- Location: Stop #3968  
- Time: Afternoon rush  
- Topic: Schedule/delay  
- Sentiment: Negative  
- Gender: Male  

This allows agencies to construct a database of rider sentiment and complaints.  

Another example:  

> *“Hey WMATA, nice trash pile to start my morning right here.”*  

This sarcastic message is classified as:  
- Mode: Train  
- Route: Green Line  
- Topic: Cleanliness  
- Sentiment: Negative (sarcasm detected)  
- Gender: Female  

Modern AI can now handle sarcasm, which older systems could not.  

---

#### Methodology  
Our system can decode unstructured data from:  
- Twitter/X  
- CRM (Customer Relationship Management) systems  
- Open-ended feedback forms  

AI extracts attributes such as:  
- Mode, route, vehicle, location  
- Complaint category (schedule, cleanliness, safety, etc.)  
- Sentiment level (positive, negative, sarcastic, neutral)  
- Demographic inferences  

---

#### Applications of the Classification System  
With structured data, agencies can:  
- Report metrics (e.g., complaints per 1,000 riders, average sentiment)  
- Monitor issues proactively  
- Detect trends in sentiment  
- Conduct comparative performance analysis  

Categories of classification include:  
- Operations  
- Fares  
- Customer service  
- Facilities and maintenance  
- Public order  
- General comments  

Each is broken into 23 subtopics, allowing agencies to assign responsibility to the right department.  

---

#### Examples of Insights  
- **Complaint classification:**  
  - A tweet about a bus running a red light is classified under *driving problems.*  
- **Trend analysis:**  
  - Station cleanliness complaints can be tracked over time.  
  - Example: Red Line showed significantly higher complaints compared to other lines.  
  - After a targeted cleaning campaign, complaints decreased—demonstrating **policy effectiveness.**  
- **Spatial analysis:**  
  - Heatmaps identify hotspots of recurring complaints.  
  - Staff can click into specific locations and view real comments.  

This tool is now deployed across WMATA, with staff using it weekly to track sentiment and feedback.  

---

#### Broader Observation: What Counts as Data?  
Traditionally, transportation agencies treated only **structured, numerical data** (e.g., ridership counts, travel times) as “real” data.  

But valuable information also exists in:  
- Text  
- Images  
- Videos  
- Audio  

For decades, agencies ignored this unstructured data because they lacked the tools to process it. With AI, we can now integrate qualitative data into operational decision-making, giving these voices the respect they deserve.  

---

#### MIT–Industry Innovation Pipeline  
At MIT, our collaborations with industry follow a cycle:  

1. **Idea phase**  
   - Work with agencies to define meaningful problems and prioritize ideas.  

2. **Methodology development**  
   - MIT researchers, graduate and undergraduate students develop innovative AI methods.  

3. **Industrialization and implementation**  
   - Agencies and consulting firms lead scale-up and deployment.  

4. **Feedback loop**  
   - MIT and partners monitor effectiveness, generating new ideas for future innovation.  

---

#### Three Levels of Contribution  
MIT contributes to industry at three levels:  

1. **Short-term, specific technical problems**  
   - Example: For Hong Kong MTR, building a stress-test system to evaluate how major events (e.g., new line openings, protests) impact service delivery.  

2. **Medium-term platforms and capabilities**  
   - Example: Developing travel demand management systems and performance models that agencies can use for many future situations.  

3. **Long-term strategic changes**  
   - Example: Working with Transport for London (TfL) after its creation in 2001.  
     - Brought together London Underground, buses, and roads under one organization.  
     - MIT helped set up TfL’s analytics team to unify data sources.  
     - Beyond technical innovation, this work strengthened TfL’s organizational identity and cohesion.  

---

#### Conclusion  
AI sentiment analysis allows agencies to tap into unstructured feedback from riders, transforming complaints and comments into actionable insights.  

By integrating these methods into a larger innovation pipeline, MIT contributes not only to solving immediate technical problems, but also to building long-term capabilities and even catalyzing institutional transformation.  