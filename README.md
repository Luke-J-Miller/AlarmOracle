# AlarmOracle
## Iterative Root Cause Discovery of Network Alarm Data using High-Precision, Low-Sensitivity Graph Neural Networks.
<!-- Centered and resized image -->
<div align="center">
  <img src="https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/a7aa422a-fcaa-43c0-a0ed-92429f80c885" alt="image_alt_text" width="200" height="200"/>
</div>


Alarm Oracle is a semi-supervised, Hawkes-process graph neural network used to conduct root cause analysis on alarm cascades.  It works iteratively by finding root causes with high confidence, adding them to the list of known root causes and beginning again. It has been specifically tailored to maintain high-precision at the expense of precision.  That is, it frequently does not identify all causasl correlations; however, the correlations it finds are incredibly likely to be true and accurate.  In practice, this is a tool that is used by domain experts to diagnose and analyze network alarm data.  Alarm Oracle works with the network administrator to identify the structures underlying alarm cascades.  The administrator then either fixes the cause of the cascade, or adds it to known root causes for future analysis.


![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/7ed3b2e1-3c86-46bb-a037-77d385cb7a60)
source: https://en.wikipedia.org/wiki/Hawkes_process

## Problem
Talk about the network causal alarm problem

First of all, the number of alarms occurring in a power system simultaneously when
a fault occurs is a crucial issue to concern. Once the Hydro-Québec Regional Control
Center reported the maximum number of alarms which could be triggered by several
types of events as follows (Durocher, 1990):
z up to 150 alarms in 2 seconds for a transformer fault;
z up to 2000 alarms for a generation substation fault, the first 300 alarms being
generated during the first 5 seconds;
z up to 20 alarms per seconds during a thunderstorm;
z up to 15000 alarms for each regional center during the first 5 seconds of a complete system collapse
Ma, T., Xiao, J., Xu, J., Guo, C., Yu, B., Zhu, S. (2011). Handling Power System Alarm Cascade Using a Multi-level Flow Model. In: Hu, W. (eds) Electronics and Signal Processing. Lecture Notes in Electrical Engineering, vol 97. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-21697-8_22



In modern control rooms, there is a constant risk of
information overload. In particular, as soon as there is a
larger disturbance, the alarm system tends to overflow
with alarms. This, in turn, makes the alarm list and the
alarm system less useful, especially in situations where it
would be most beneficial. In fact, when an incident starts,
operators tend to disregard the alarm list, and only when
the incident has been taken care of, they return to the list,
remove all waiting alarms, and then start using it for
monitoring. This effect seems to be more common the
better equipped the systems is with alarms, and is well
known in nuclear control rooms.
Larsson, J. E., Oehman, B., Calzada, A., Nihlwing, C., Jokstad, H., Kristianssen, L. I., Kvalem, J., & Lind, M. (2006). A revival of the alarm system: Making the alarm list useful during incidents. United States: American Nuclear Society - ANS.




Every day, tens of thousands of faults are triggered across heterogeneous and interconnected devices in a telecom network. These faults are expressed by the network devices in the form of alarms, which are transmitted to the Network Operation Centre (NOC) for further processing by network operators. Additionally, there are thousands of types of alarms. If the network operators handle all alarms sequentially, they will be overloaded and unable to concentrate on finding the underlying reasons for the faults. Generally, approximately 1 million alarms are reported to the NOC every day. If there are five network operators working 8 h per day, each operator must process 20 alarms every minute throughout the day, which is an impossible workload. Therefore, it is necessary to select the important alarms that are useful for identifying network problems.
Jiantao Wang, Caifeng He, Yijun Liu, Guangjian Tian, Ivy Peng, Jia Xing, Xiangbing Ruan, Haoran Xie, Fu Lee Wang,
Efficient alarm behavior analytics for telecom networks,
Information Sciences,
Volume 402,
2017,
Pages 1-14,
ISSN 0020-0255,
https://doi.org/10.1016/j.ins.2017.03.020.
(https://www.sciencedirect.com/science/article/pii/S0020025517306059)


## Data
Talk about modeling the procedural data generation after real world networks.

## Model
Temporal Process Miner with Hawkes processes

## Results
Some additional points.  Alarms and any alrams they trigger are probabilistic.  The chcance of an alarm occuring in any second on any device is 1e-8.  Additionally, an alarm cannot reach across the network if the subsequent alarm that can be triggered is not on the same device or a neighboring device.  Thus, while there exists causal relationships in the alarm matrix, it is possible for such an alarm to not be in the data.  This could be either because the time window didn't capture sufficient evidence of the event, or the topology precludes it.
![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/736cf6ab-cfbf-4977-9012-7a3ffd3874f1)

![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/99259274-4555-4ca1-82a7-9acf2e857f42)

![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/a2a13437-6995-4bb1-a4d4-ac953e1f1a9c)

![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/497e32fa-b4e2-4798-8ea4-333a6ed689c2)




## Real world Application
;alskdjfa;lskdfjk

## Future work
This section needs heavilty stresseed that this is preliminary work.  In fact, we need to express this throughout the slideshow and readme.  Future work could include comparing device topologies with alarm data and information to eliminate possibilities prior to analysis.  Another method would be to have multiple windows of time from the same topology.  An analysis is ran, and some high confidence causalities are added to bolster the prior information given to the model.
