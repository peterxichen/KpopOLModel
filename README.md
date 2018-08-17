# KpopOLModel
Statistical model of the South Korean idol industry and simulations applying optimal learning techniques to maximize profits for entertainment agencies. Final project for ORF 418.

Deployment
----------------------------------------------
We use a belief model consisting of an outer lookup table and inner parametric model, and test different combinations of policies to determine the best one:

- myPE_PE: pure exploitation lookup table, pure exploitation parametric
- myIE_IE: interval estimation lookup table, interval estimation parametric
- myKG_KG: knowledge gradient lookup table, knowledge gradient parametric
- myPE_IE: pure exploitation lookup table, interval estimation parametric
- myPE_KG: pure exploitation lookup table, knowledge gradient parametric
- myIE_PE: interval estimation lookup table, pure exploitation parametric
- myIE_KG: interval estimation lookup table, knowledge gradient parametric
- myKG_PE: knowledge gradient lookup table, pure exploitation parametric
- myKG_IE: knowledge gradient lookup table, interval estimation parametric

- myRun.m generates the truth values and executes other scripts
- myTuning.m tunes paramters for interval estimation policies

Authors
----------------------------------------------

- Peter Chen
- Tor Nitayanont
