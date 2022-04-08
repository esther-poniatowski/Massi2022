<div id="top"></div>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]

<h3 align="center">Massi 2022 - Model-based and model-free replay mechanisms for reinforcement learning in neurorobotics</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#contributors">Contributors & Contacts</a></li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>


## About the Project

> This repository is related to the article :  
> **Model-based and model-free replay mechanisms for reinforcement learning in neurorobotics** (2022, Submitted)   
> Elisa Massi, Remi Dromnelle, Julianne Mailly, Jeanne Barthéléemy, Julien Canitrot, Esther Poniatowski, Benoît Girard and Mehdi Khamassi.   
> _Institute of Intelligent Systems and Robotics, CNRS, Sorbonne University, F-75005_  
> _Paris, France_  
> It contains codes and data used and generated for the part :  
> **3 Simulation of individual replay strategies with an autonomously learned state decomposition**  
Keywords: `hippocampal replay`, `reinforcement learning`, `neurorobotics`, `model-based`, `model-free`

Project Link: [https://github.com/esther-poniatowski/Massi2022](https://github.com/esther-poniatowski/Massi2022)

### Goals of the modeling 

To study the implications of offline learning in spatial navigation, from rodents' behavior to robotics, this article investigated the role of several Reinforement Learning (RL) algorithms, by simulating artificial agents. 
The task of the agents mimicks the classical [Morris water maze task](http://www.scholarpedia.org/article/Morris_water_maze) [(Morris, 1981)](https://www.nature.com/articles/297681a0). The environment is defined by a circular maze, consistent with the original experimental paradigm in terms of environment/robot size ratio. The goal of the task is to navigate the environment until reaching the rewarded location, starting from a fixed initial point. Agents learn over 50 trials, and the reward location is changed at the middle of the simulation (trial 25). In this robotic framework, the task is a Markov decision problem (MDP), where agents visit discrete states, using a finite set of discrete actions.

The learning performances of the agents are tested here in two conditions:
- *Deterministic environment*: In this version of the task, any action a performed in a given state always leads the agent to the same arrival state (with probability 1).
- *Stochastic environment*: In this version of the task, performing action in a given state can lead to distinct possible arrival state (non-null probabilities for several states).

Four learning strategies are compared. Three of them include replays of the experienced state-action-state transitions during each inter-trial interval.
- *Model Free (MF) No replay*: In this classical reinforcement learning framework, the artificial agent learns only online, during behavior.
- *Model Free (MF) Backward replay*: This agents stores the most recent experienced state-action-state transitions in a memory buffer, and replays them from the more recent (rewarded) one to the most remote one.
- *Model Free (MF) Backward replay*: This agents stores the most recent experienced state-action-state transitions in a memory buffer, and replays them in random order.
- *Model Based (MB) Prioritized sweeping*: This agents stores the most recent experienced state-action-state transitions in a memory buffer, and replays them from the more recent (rewarded) one to the most remote one.
*Note that one more replay strategy (Most diverse sequence replay) appears in the code, but is not investigated in the related article.*

### Contributors & Contacts

- [Elisa Massi](https://github.com/elimas9) - massi@isir.upmc.fr
- [Esther Poniatowski](https://github.com/esther-poniatowski) - eponiatowski@clipper.ens.psl.eu
- [Juliane Mailly](https://github.com/julianemailly) 

<p align="right">(<a href="#top">back to top</a>)</p>


## Usage

### Architecture of the project

The project is made up of the following files and directories :
- [ ] Two Jupyter notebooks guide the execution of the main functionalities. 
  - `Navigation_generate_data.ipynb` can be used to generate data, with arbitrary parameters and different versions of the task.
  - `Navigation_alanysis.ipynb` provides graphical visualization of the results, reproducing in particular the figures of the article.
- [ ] Nine python files correspond to the [modules](### Modules) called by the Jupyter notebooks.
- [ ] The folder `Data/` is the location where generated data are stored. called by the Jupyter notebooks.
  - It aready contains most of the data files required to plot the figures from the Jupyter notebooks. Files' formats are either `.csv` (for dataframes) or `.pickle` (for dictionaries, arrays, lists).
  - The sub-folder `Data_indiv/` specifically contains detailed data for 100 individual artificial agents.
- [ ] The folder `Figures/` is the location where generated figures can be saved. It already contains the file `map1.pgm` necessary to plot one type of figure, representing the environment.

### Requirements

All codes are built in Python 3.
The following libraries are used:
- numpy
- random
- itertools
- collections
- copy
- scipy
- bioinfokit
- statsmodels
- similaritymeasures
- pandas
- pickle
- matplotlib
- seaborn

### Modules

All the modules only contain functions (no script), which are called by the Jupyter notebooks.  
More details about those modules and functions are available in the code documentation (accessed via `help()`).

- [ ] `parameters_MF_MB.py` - Defines the parameters of the simulation. All the parameters are collected in a dictionary, which is provided to the main functions as a default argument (as the module is imported in the preamble of all other files).
- [ ] `algorithms_MF_MB.py` - Implements the reinforcement learning procedure and the different replay strategies, necessary to perform one trial (behavior + replay).
- [ ] `simulations_MF_MB.py`- Generates simulations of `n_individuals` (100) agents over `n_trials` (50) trials, in a given environmental condition (deterministic/stochastic). Saves data in the appropriate folder.
- [ ] `analyzes_MF_MB.py` - Extract relevant features of the data: computes summary statistics, performs statistical analyses...
- [ ] `figures_MF_MB` - Generates the main functions of the article.
- [ ] `figures_indiv`, `figures_pop`, `figures_qvalue_map`, `figures_utils` - Other graphical functions to display results more flexibly in exploratory invesigations.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[product-screenshot]: images/screenshot.png
