# S2DTA
 S2DTA is a graph-based deep learning model for predicting drug-target affinity (DTA) by fusing sequence and structural knowledge. 

 The benchmark dataset can be found in ./data/. The pretrained models are available in ./data/pretrained_models/. The S2DTA models is in ./. For more details, please read our paper.
 
 # Requirements
 - python 3.9
- numpy 1.24.0
- biopython 1.8
- torch 1.13.1
- cudnn 11.6
- scikit-learn 1.2.0
- scipy 1.9.3
- pandas 1.5.2
- torch-geometric 2.2.0
 
 # Traing and testing
 
 **In this module you have to provide .pdb file for protein and pocket, .sdf or .mol file for compund.**
if you want to training the model with your data
`python create_data.py`
`python training.py`
only use our model to evaluat your own data
`python create_data.py`
`python run_pretrained_model.py`
 
 # Contact
 Xin Zeng: hbzengxin@163.com
