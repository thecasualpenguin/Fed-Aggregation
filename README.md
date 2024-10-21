# Fed-Aggregation

Paper: Communication-Efficient Learning of Deep Networks from Decentralized Data [ICML'17].

Reprosued some of the MNIST experiments from the seminial paper of McMahan et al., 2017.

To run experiments, see the notebook fed_avg.ipynb.

See fed_avg.pdf for full experimental details and results.

Below are plots of test accuracy after t rounds of FedAvg:

for the iid and non-iid data setup;
for a CNN vs a 2 hidden-layer MLP (2NN);
for selecting m=10 or 50 clients each round.
iid noniid

E refers to the number of epochs for local training, for each client, for each round.

