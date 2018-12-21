# SGPLVM-JCP-Experiments
Experiments using the structured Bayesian Gaussian process latent variable model for inverse problems.

See [Atkinson and Zabaras (2018)](#)

## Prerequisites

This repository uses python >=3.5 and [structured-gpflow](https://github.com/cics-nd/structured-gpflow.git).
You can try running the included `configure.bash` to set things up quickly.

## Running the experiments

The main experiment modules are:
* `elliptic_forward_sgplvm_separate.py` 
(Forward problem with two-model approach)
* `elliptic_forward_sgplvm_joint.py` 
(Forward problem with jointly-trained model)
* `elliptic_inverse_sgplvm_separate.py` 
( Inverse problem with two-model approach)
* `elliptic_inverse_sgplvm_joint.py`
(Inverse problem with jointly-trained model)

bash scripts that help with suggested parameter settings from the paper can be found in the `scripts` directory.

## Data and plotting

Data used for the experiments in the paper is provided in `elliptic/data`.
The MATLAB script for plotting the files outputted by the experiements can be 
found at `ellitpic/matlab/plot_predictions.m`.

Any questions, email [Steven Atkinson](steven@atkinson.mn) or 
[Nicholas Zabaras](nzabaras@nd.edu)
