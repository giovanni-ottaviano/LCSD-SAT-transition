# Satisfiability transition in the linear classification of structured data 

## Overview
This repository contains part of the code I developed during my Master thesis in Theoretical Physics at UNIMI (Milan) in the field *Statistical Physics of Machine Learning*.

The goal of the research project was to investigate the linear classification of structured data, such as doublets (couple of points) with fixed overlap, and analyse the satisfiability transition undergone by the model using computational tools.

Two different frameworks are proposed, through appropriate reformulations of the problem:
- **Constraint Satisfaction Problem (CSP)**: Inspected using a suitable optimizer (Gurobi) for QCQP problems
- **Function minimization**: Examined using Root Finding algorithms and Stochastic Gradient Descent


## Usage
The folder ***src*** contains the main source code for this project, separated by specific folders:
- ***test_generator***: A C++ script to test the behaviour of the doublets' generator
- ***qcp***: A script to solve the CSP using with a specific optimizer (requires Gurobi)
- ***minimization***: A set of scripts to run the function minimization (one per method)

Every run can be customised by giving a specific set of input parameters.

The ***results*** folder contains a sample of results for the different methods with multiple sets of input parameters.

The ***visualization*** notebook includes some interesting graphical results, very helpful to gain additional insights on the problem and results.


## Bibliography
[1] E. Gardner. "Maximum Storage Capacity in Neural Networks". In: *Europhysics Letters (EPL)* (1987). DOI: 10.1209/0295-5075/4/4/016. URL: https://dx.doi.org/10.1209/0295-5075/4/4/016

[2] P. Rotondo, M. Cosentino Lagomarsino, M. Gherardi. "Counting the learnable functions of geometrically structured data".  In: *Physical Review Research* 2 (2 May 2020). DOI: 10.1103/physrevresearch.2.023169. URL: http://dx.doi.org/10.1103/PhysRevResearch.2.023169

[3] P. Rotondo, M. Pastore, M. Gherardi. : "Beyond the Storage Capacity: Data-Driven Satisfiability Transition". In: *Physical Review Letters* 123 (2020). DOI: 10.1103/physrevlett.125.120601. URL: http://dx.doi.org/10.1103/PhysRevLett.125.120601


## Contacts
For comments or questions about this project, feel free to send me an [email](mailto:giovanni.ottaviano@live.com).