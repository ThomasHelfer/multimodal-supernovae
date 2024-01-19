# IAIFI Astro Hackathon: Simulation-Based Inference

Simulation-based inference (SBI) is a powerful class of methods for performing inference in settings where the likelihood is computationally intractable, but simulations can be realized via forward modeling. 

- [Notebook 1](./notebooks/01_implicit_likelihoods.ipynb): Introduce the notion of an implicit likelihood, and how to leverage it to perform inference. Look at a "traditional" method for likelihood-free inference, Approximate Bayesian Computation (ABC);
- [Notebook 2](./notebooks/02_neural_sbi.ipynb): Build up two common modern _neural_ SBI techniques: neural likelihood-ratio estimation (NRE) and neural posterior estimation (NPE). Introduce the concept of statistical coverage testing and calibration.
- [Notebook 3](./notebooks/03_galaxies.ipynb): Open-ended problem, working with galaxy clustering simulations for parameter estimation. Learn to use more sophisticated feature-extractors for different data types.
