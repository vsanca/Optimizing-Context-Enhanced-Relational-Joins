# Optimizing Context-Enhanced Relational Joins

Code and material for Optimizing Context-Enhanced Relational Joins paper (submission ID 1273).

Disclaimer: per artifact request for every paper: "We do not expect a fully polished submission in terms of automatically reproducing results, but rather a reasonably clean version of the state of the code when submitting the paper." we provide the code.

We provide the code as-is, and the code in its current state in the repository might not be immediately and easily usable for reproducing each experiment due to the following reasons:

- the code and install scripts would require further modifications to make it immediately usable (dependency lists, correct automated scripts),
- the experiments from the paper might not be directly set up in the code, as-is, 1:1 as separate scripts to run (but with some modifications, the code was used to run the experiments, e.g., with correct parameters),
- there are hardware, platform, and kernel requirements that may not be commonly available at this point for all the experiments (the experimental setup is explicitly outlined in the paper).

Still, this is the bulk of the code used to obtain the results. Datasets, models, full experimental setup, and other necessary details (such as compiler versions or non-standard setup) are outlined and specified in the paper and, where appropriate, are included in the code from their publicly available sources to download (e.g., FastText library or some of the related models available on respective public websites or repositories).

