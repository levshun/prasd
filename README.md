# The PrASD Framework Repository

![An overviwew of the framework](img/framework.jpg)

This repository has been developed to support the research paper titled 
"PrASD: Intelligent Multi-Agent Framework for Predictive-Based Abnormal State Detection in Smart Factory Devices",
which is submitted for consideration at The ACM Conference on Computer and Communications Security (ACM CCS 2026).

# Experimental datasets

In this work, we conducted experiments using two datasets:
* AI4I-2021: Condition Monitoring Dataset, available at Kaggle: 
https://www.kaggle.com/datasets/stephanmatzka/condition-monitoring-dataset-ai4i-2021.
* CASPER: Context-Aware Anomaly Detection System for Industrial Robotic Arms, available at IEEE DataPort: 
https://ieee-dataport.org/documents/casper-context-aware-anomaly-detection-system-industrial-robotic-arms.

# Experimental setup

All experiments with ```SCAg``` conducted using Python ```v3.10``` in the ```Ubuntu WSL``` environment.
The list of required packages for reproducing those experiments provided in the ```scag_requirements.txt``` file.

All experiments with ```SPAg``` and ```ADAg``` conducted using Python ```3.10``` in the ```Windows 11``` environment.
The list of required packages for reproducing those experiments provided in the ```spag_requirements.txt``` file.

# Experimental evaluation

All artifacts required to reproduce the paper's contribution for the AI4I-2021 dataset are available in 
the ```ai4i``` folder, while those for the CASPER dataset are located in the ```casper``` folder.

To facilitate usability, we have prepared the following Jupyter Notebooks:
* ```scag_demo.ipynb```: reproduces all evaluations for ```C-SCAg``` (Current State Classification Agent) 
and ```P-SCAg``` (Predicted State Classification Agent). 
* ```spag_demo.ipynb```: reproduces all evaluations for ```SPAg``` (State Prediction Agent). 
* ```adag_demo.ipynb```: reproduces all evaluations for ```ADAg``` (Anomaly Detection Agent). 

Please note that the notebooks for both datasets share the same filename but are stored 
in their respective folders (```ai4i``` or ```casper```).

Other folders include:
* ```models```: contains pretrained model checkpoints (```*.pt``` and ```*.keras``` files) and parameter files 
(```*.json```). The naming convention for models corresponds to that  used in the paper. Note that the 
```models``` directory includes subfolders associated with the framework modules; for example, 
the ```scag``` subfolder contains pretrained models for  ```C-SCAg``` and ```P-SCAg```.
* ```data```: holds the data utilized for model training, validation, and testing for different agents, with respective
subfolders naming scheme – ```c-scag``` for real-world data with labels, ```p-scag``` for predicted data with labels, 
etc. Data is stored in ```*.npy``` and ```.csv``` files, with features (```X```) and labels (```y```) separated. 
Most of the files are presented in resampled version to comply with repository size limitations.
* ```adag```: contains the model class ```AnomalyDetectorPro``` for ```ADAg```.
* ```scag```: contains the model classes for ```C-SCAg``` and ```P-SCAg```, including ```ParametrizedMLP```, 
```ParametrizedCNN```, ```ParametrizedRNN``` and ```ParametrizedTransformer```.
* ```spag```: contains the model classes for ```SPAg```, including ```BaseForecaster``` and
```DeepForecaster```.
* ```support```: contains additional library files for ```ADAg```, ```SCAg``` and ```SPAg```.

It is important to note that the datasets ```c-scag``` and ```p-scag``` comprise files containing data 
processed after the wavelet transformation. Among the available options, the ```sym8``` wavelet function 
in ```symmetric``` mode was selected, as it demonstrated the most robust and consistent results 
across all conducted experiments.