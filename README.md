**This is not an officially supported Google product.**


# Learning Decoupled Local and Global Representations for Time Series

![overview](https://user-images.githubusercontent.com/93283484/140096231-4a1c18c3-8a12-450d-b74a-1abcce2528fe.jpg)


### Cloning the Conda environment
Use the env-file.txt to recreate the conda environment with all the libraries required for the experiments. 
```
conda create --name tf-gpu --file env-file.txt
```


### Training the GLR model
You can use the main script to train the local and global representation learning model. This script will train all the model components, and plot the distribution of the local and global representation for the population.
```
python -m main --data [DATASET_NAME] --lamda [REGULARIZATION_WEIGHT] --train
```


### Training baseline models
In order to replicate the baseline experiments, train the baseline models (VAE or GPVAE) using the following script
```
python -m baselines.vae --data [DATASET_NAME] --rep_size [Z_SIZE] --train

python -m baselines.gpvae --data [DATASET_NAME] --rep_size [Z_SIZE] --train
```

### Running the evaluation tests
All codes for the evaluation experiments can be found under the evaluations directory. Make sure to specify the baseline model in the code when running each experiment.
