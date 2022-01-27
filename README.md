**This is not an officially supported Google product.**


# Learning Decoupled Local and Global Representations for Time Series

![overview](https://user-images.githubusercontent.com/93283484/140096231-4a1c18c3-8a12-450d-b74a-1abcce2528fe.jpg)


### Training the GLR model
You can use the main script to train the local and global representation learning model for the datasets presented in the paper. This script will train all the model components, and plot the distribution of the local and global representation for the population. 
```
python -m main --data [DATASET_NAME] --lamda [REGULARIZATION_WEIGHT] --train
```

#### In order to train the GLR model on your own dataset, follow the steps below:

1. Create your encoder and decoder architectures and instantiate a GLR model for your dataset
```
glr_model = GLR(global_encoder, local_encoder, decoder, time_length, data_dim, window_size, kernel, beta, lamda)
```
3. Create your own data loader function and dataset object (Make sure it includes the sample, mask, and the sample length)
4. Train the model!
```
glr_model.train(trainset, validset, [NAME OF DATASET], lr, n_epochs)
```


### Training baseline models
In order to replicate the baseline experiments, train the baseline models (VAE or GPVAE) using the following script
```
python -m baselines.vae --data [DATASET_NAME] --rep_size [Z_SIZE] --train

python -m baselines.gpvae --data [DATASET_NAME] --rep_size [Z_SIZE] --train
```

### Running the evaluation tests
All codes for the evaluation experiments can be found under the evaluations directory. Make sure to specify the baseline model in the code when running each experiment.
