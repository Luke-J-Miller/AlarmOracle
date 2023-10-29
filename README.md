# AlarmOracle
## Iterative Root Cause Discovery of Network Alarm Data using High-Precision, Low-Sensitivity Graph Neural Networks.
<!-- Centered and resized image -->
<div align="center">
  <img src="https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/a7aa422a-fcaa-43c0-a0ed-92429f80c885" alt="image_alt_text" width="200" height="200"/>
</div>


Alarm Oracle is a semi-supervised, Hawkes-process graph neural network used to conduct root cause analysis on alarm cascades.  It works iteratively by finding root causes with high confidence, adding them to the list of known root causes and beginning again. It has been specifically tailored to maintain high-precision at the expense of precision.  That is, it frequently does not identify all causasl correlations; however, the correlations it finds are incredibly likely to be true and accurate.  In practice, this is a tool that is used by domain experts to diagnose and analyze network alarm data.  Alarm Oracle works with the network administrator to identify the structures underlying alarm cascades.  The administrator then either fixes the cause of the cascade, or adds it to known root causes for future analysis.


![image](https://github.com/Luke-J-Miller/AlarmOracle/assets/111100132/7ed3b2e1-3c86-46bb-a037-77d385cb7a60)
source: https://en.wikipedia.org/wiki/Hawkes_process

