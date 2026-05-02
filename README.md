# The PrASD Framework Repository

![An overviwew of the framework](img/framework.jpg)

This repository has been developed to support the research paper titled 
"PrASD: Intelligent Multi-Agent Framework for Predictive-Based Abnormal State Detection in Smart Factory Devices",
which is submitted for consideration at The ACM Conference on Computer and Communications Security (ACM CCS 2026).

# Experimental datasets

In our work we experimented with two datasets:
* AI4I-2021: Condition Monitoring Dataset, available at Kaggle: https://www.kaggle.com/datasets/stephanmatzka/condition-monitoring-dataset-ai4i-2021.
* CASPER: Context-Aware Anomaly Detection System for Industrial Robotic Arms, available at IEEE DataPort: https://ieee-dataport.org/documents/casper-context-aware-anomaly-detection-system-industrial-robotic-arms.

# Experimental evaluation

All artifacts required to reproduce the paper's contribution for the AI4I-2021 dataset are available in the ```ai4i``` folder, 
while those for the CASPER dataset are located in the ```casper``` folder.

To facilitate usability, we have prepared the following Jupyter Notebooks:
* ```cscag_demo.ipynb```: reproduces all evaluations for ```C-SCAg``` (Current State Classification Agent) 
and ```P-SCAg``` (Predicted State Classification Agent). Please note that the notebooks for both datasets share 
the same filename but are stored in their respective folders (```ai4i``` or ```casper```).

Other folders include:
* ```models```: contains pretrained model checkpoints (```*.pt``` files) and parameter files (```*.json```). 
The naming convention for models corresponds to that used in the paper. Note that the ```models```directory includes 
subfolders associated with the framework modules; for example, the ```scag``` subfolder contains pretrained models 
for  ```C-SCAg``` and ```P-SCAg```.
* ```real_data```: holds the data utilized for model training, validation, and testing on real-world data. 
Data is stored in ```*.npy``` files, with features (```X```) and labels (```y```) separated. 
If filenames use the prefix ```Xr_``` | ```yr_``` instead of ```X_``` | ```y_```, 
this indicates a resampled version of the dataset, provided to comply with repository size limitations.
* ```pred_data```: contains data employed for model evaluation on predicted datasets, also in ```*.npy``` format 
with separated features (```X```) and labels (```y```). The filename prefix specifies the model used for prediction 
(e.g., ```GRU-1_``` indicates that the ```GRU-1``` model was used for data prediction), following the same naming 
scheme as in the paper. The meaning of ```_Xr_``` | ```_yr_``` is consistent with that in the ```real_data``` folder.

The files defining the model architectures are separated and stored in the root directory:
* ```scag_model.py```: contains the model classes for ```C-SCAg``` and ```P-SCAg```, including ```ParametrizedMLP```, 
```ParametrizedCNN```, ```ParametrizedRNN``` and ```ParametrizedTransformer```.
