# Neural Network Trainer

Please see example.yaml for a list of supported options and arguments

### main.py:

Based on the input arguments provided in the yaml file, main.py will instantiate the appropriate a trainer and model classes and train the neural network model according to these specifications. 

If multiple values are provided for a single argument this script will take the cartesian product of all 
arguments and train a single network for each unique permutation.

### Trainers.py:

Trainer - Implements data loading, forward pass, backpropagation and plotting

Inherited by:

ObservationTrainer > Train basic vision models.
Saccadetrainer > Train 'saccading' vision models.
Transitiontrainer > Train dynamic models of time series / videos
PrednetTrainer > Train Prednet (encode prediction errors)
InteractiveTrainer > Interact with Unity during trainer

Each trainer optimises a single 'model'.

### Models.py:

Implements the data flow in the forward pass of a model. 

ObservationVAE > Simple VAE
VRNN > Variational recurrent neural network (simple world model)
HVRNN > Hierarchical variational recurrent neural network (https://arxiv.org/pdf/1904.12165.pdf)
PrednetWorldModel > based on https://arxiv.org/abs/1803.10122 and https://coxlab.github.io/prednet/

Each model consists of one or more 'modules'.

### Modules.py:

Defines requires tensor operations. 

Provides encoding and decoding networks for the MNIST, Celeba, animalai, stl-10, dSprites, atari.
Includes conv, FC and resnet versions of most of these. 