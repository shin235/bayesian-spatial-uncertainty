# Spatial Uncertainty Quantification in DeepKriging: A Bayesian Approach

This repository was developed as part of my Master’s thesis at Humboldt-Universität zu Berlin.

It implements a probabilistic deep learning pipeline to predict PM2.5 concentrations across the United States. The model builds on Bayesian deep learning techniques to quantify predictive uncertainty.

The datasets and preprocessing in the `data` directory follow the implementation by Chen et al. (2022), _DeepKriging: Spatially Dependent Deep Neural Networks for Spatial Prediction_ ([GitHub](https://github.com/aleksada/DeepKriging)).

To better capture spatial uncertainty, I propose a Bayesian approach. The input is divided into two branches: one for spatial basis functions (phi branch) and another for meteorological covariates (covariate branch). Bayesian layers are applied in the early phi branch and again after merging the two branches, allowing the model to estimate uncertainty arising from spatial representation.
