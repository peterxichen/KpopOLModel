# KpopOLModel
Statistical model of the South Korean idol industry and simulations applying optimal learning techniques to maximize profits for entertainment agencies. Final project for ORF 418.

Abstract
----------------------------------------------
The Korean popular music industry, K-pop for short, is characterized by major entertainment agencies that manage all aspects of their signed artists' professional careers. Known as "idols," these musical artists receive extensive training for several years in singing, rapping, dancing, acting and/or languages before debuting, usually in a group. In our problem, we are the company, and our goal is to maximize profits out of an idol. In 2009, South Korea's Fair Trade Commission imposed a limit of seven years to entertainment contracts. As a result, a major learning problem that faces each entertainment agency is the optimal use of their artists' schedules to maximize profits over the course of the contract period. We simplify the problem by generalizing four categories of activities for each time period of 3 months: producing an album, touring, acting, or recording variety shows.

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
