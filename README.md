# DeepLearning_Project
Transformer applied in Skeleton-based human action recognition


## Getting start

0) download the projet via git:
 >     git clone https://github.com/Annelise2019/DeepLearning_Project.git

1) download the dataset NTU-60 via:
      http://rose1.ntu.edu.sg/datasets/actionrecognition.asp  
      
   And then, this command should be used to build the database for training or evaluation:
 >     python tools/ntu_gendata.py --data_path <path to nturgbd+d_skeletons>

2) change the data path in these files
    data_process/skeleton_feeder.py
    config.py 
3) change the gpu setting
    config.py
  
4) run the model using this command:
>      python classify.py

6) result: 
a folder named "log" will be created to store all the logging traces of your tranning  
a folder named "checkpoint" will be created to store the model parameters and thus you can retrain your model by changing relative parameters in config.py
          
