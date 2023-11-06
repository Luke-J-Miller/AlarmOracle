# AlarmOracle

## Iterative Root Cause Discovery of Network Alarm Data using High-Precision, Low-Sensitivity Graph Neural Networks.
<!-- Centered and resized image -->
<div align="center">
  <img src="https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/a7aa422a-fcaa-43c0-a0ed-92429f80c885" alt="image_alt_text" width="200" height="200"/>
</div>


Alarm Oracle is a semi-supervised, Hawkes-process graph neural network used to conduct root cause analysis on alarm cascades.  It works iteratively by finding root causes with high confidence, adding them to the list of known root causes and beginning again. It has been specifically tailored to maintain high-precision at the expense of sensitivity.  That is, it frequently does not identify all causasl correlations; however, the correlations it finds are incredibly likely to be true and accurate.  In practice, this is a tool that is used by domain experts to diagnose and analyze network alarm data.  Alarm Oracle works with the network administrator to identify the structures underlying alarm cascades.  The administrator then either fixes the cause of the cascade, or adds it to known root causes for future analysis.


## Problem
Every day, administrator of complex networks are faced with a deluge of alarms. In some instances an alarm cascade occurs.  An alarm cascade is an event where one alarm or fault cases a massive amount of subsequent alarms.  These alarms were put in place to alert operators to a failure, but when an alarm cascade occurs, it can be nearly impossible to track down the root cause.
### Real examples of disasters caused by alarm floods and cascades.
#### Three Mile Island Nuclear Accident (1979)
The Three Mile Island incident is perhaps one of the most cited cases when discussing the dangers of alarm floods. A reactor at the Three Mile Island plant near Harrisburg, Pennsylvania, suffered a partial meltdown on March 28, 1979. After a series of mechanical failures, including a stuck valve and cooling malfunctions, operators were bombarded with over 100 alarms in a short period. These alarms came in the form of lights, sounds, and gauges showing abnormal conditions.

The inundation of alarms overwhelmed the operators and severely hampered their ability to diagnose the situation accurately. The cacophony of alarms led to sensory overload, making it difficult to prioritize responses or even understand what was going wrong. While other human and mechanical factors also contributed, the alarm cascade effectively delayed quick and precise intervention, leading to the release of a small amount of radioactive gases.

#### Piper Alpha Oil Platform Disaster (1988)
Another case where alarm cascades played a role was the Piper Alpha disaster in the North Sea. On July 6, 1988, a series of explosions and fires destroyed the Piper Alpha oil platform, resulting in 167 deaths. A gas leak initiated a fire, which then triggered a series of additional events and alarms. The alarm systems were not designed to handle such a high level of complexity and interdependence between various systems.

Operators were inundated with alarms but had no effective way to prioritize them. The situation was made worse by the absence of an automatic shutdown system for certain parts of the platform. The crew was overwhelmed, and emergency procedures proved to be woefully inadequate, partly because of the confusion caused by the numerous alarms.

#### Texas City Refinery Explosion (2005)
In this case, a BP oil refinery in Texas City exploded, killing 15 workers and injuring over 180 others. While human errors and neglect of safety procedures were the primary causes, the control room was subjected to multiple alarms without clear prioritization. Operators failed to notice key indicators of an impending disaster, partially due to the high number of alarms that they had become conditioned to ignore. This phenomenon, often called "alarm fatigue," led to delays in taking preventive action.

#### Healthcare and Alarm Fatigue
In modern healthcare settings, especially intensive care units, alarm fatigue has become a significant problem. Medical staff are exposed to a plethora of alarms from various equipment like heart rate monitors, ventilators, and infusion pumps. The frequent alarms desensitize the healthcare workers, making it more likely for them to miss or delay responses to critical alarms. This has led to preventable adverse events, although these are typically not as dramatic as large-scale industrial accidents.

#### Industrial Internet of Things (IIoT)
In the context of the Industrial Internet of Things (IIoT), alarm cascades could potentially become a significant issue. As industrial settings become more interconnected, the risk of an alarm flood that could overwhelm operators grows. While not a specific incident, it's an area of concern that is gaining attention.

## Data
The data used for training and evaluating the dataset was procedurally generated.  We had access to a dataset from Huawei; however, it did not have ground truth information as to what alarm caused the event.  So, we generated our own data to match the distribution of the Huawei data.  An artificial network of devices was created.  For each device in the network, there were possible alarms assigned with a probability of an alarm occuring at any time.  Additionally, some alarms had a probability of triggering another alarm on the same device or a device adjacent to it.  This topology of devices along with these probabilistic alarms was used to generate a log of alarms for training.

## Model
### AlarmOracle - A Hybrid Approach to Temporal Causal Discovery
#### Introduction:
AlarmOracle aims to discover causal relationships in event-based temporal networks using a hybrid approach that combines graph theory, statistical learning, and optimization techniques. This enriched framework encapsulates NetworkX graphs, pandas DataFrames, Expectation-Maximization (EM), Hawkes processes, Poisson processes, and optimization methods.

#### Key Structures:
Poisson Processes: These are employed for the basic modeling of random events in the network when no historical events are considered.
Hawkes Processes: Extends Poisson processes by including the memory of past events to predict the conditional intensity of future events.
#### Modalities:
Stochastic Processes: The algorithm integrates Hawkes and Poisson processes for a nuanced understanding of temporal event dynamics.
Optimization Techniques: Employs gradient descent and hill climbing to optimize hyperparameters and causal weights.
Algorithmic Steps:
### Initialization:

Utilizes Poisson processes to provide initial estimates for events that do not have prior occurrences.
#### Hop-Analysis:

Applies Dijkstra's algorithm to map the shortest paths between nodes, and to determine k-hop neighbors.
#### Hawkes Process Modeling:

Utilizes the memory function to model the intensity of events as influenced by previous events.
#### Expectation-Maximization (EM):

Employs Hawkes processes within the EM algorithm to better fit the event data and update causal matrices.
#### Gradient Descent & Hill Climbing:

Gradient descent is used for continuous optimization of causal impact weights.
Hill climbing is used for discrete optimization problems, such as choosing the optimal number of 'k-hops'.
#### Stopping Criteria:

Uses Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC) along with the performance of gradient descent and hill climbing to decide when to stop iterations.
Complexity :
The addition of Hawkes processes and optimization techniques slightly increases the computational complexity but enhances the algorithm's predictive performance.

## Results
Some additional points.  Alarms and any alrams they trigger are probabilistic.  The chcance of an alarm occuring in any second on any device is 1e-8.  Additionally, an alarm cannot reach across the network if the subsequent alarm that can be triggered is not on the same device or a neighboring device.  Thus, while there exists causal relationships in the alarm matrix, it is possible for such an alarm to not be in the data.  This could be either because the time window didn't capture sufficient evidence of the event, or the topology precludes it.
### Precision
![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/736cf6ab-cfbf-4977-9012-7a3ffd3874f1)

![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/99259274-4555-4ca1-82a7-9acf2e857f42)

![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/a2a13437-6995-4bb1-a4d4-ac953e1f1a9c)

![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/497e32fa-b4e2-4798-8ea4-333a6ed689c2)


### Time Complexity  
O(N log N) (for alarm data in start_init)} +   
O(|E| + |V| log |V|) (for topology in k_hop_neighbors) +   
O(max_iter * N^3) for alarm data in hill_climb and em) +   
O(k * N) (for num hops (k)  and alarm data _get_effect_tensor_decays() & _get_effect_tensor_decays_each_hop()  
The likely dominating term i O(N^3) where N is the number of logged events.  
This is somewhat mitigated by the data preprocessing. 

## Real world Application
AlarmOracle as a proof of concept has shown remarkable results, and the team looks forward to finding application in real-world envirionments.

## Future work
Future work could include comparing device topologies with alarm data and information to eliminate possibilities prior to analysis.  Another method would be to have multiple windows of time from the same topology.  An analysis is ran, and some high confidence causalities are added to bolster the prior information given to the model.
