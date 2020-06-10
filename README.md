# DisCont: Self-Supervised Visual Attribute Disentanglement using Context Vectors

This repository contains code for the paper <a href="">DisCont: Self-Supervised Visual Attribute Disentanglement using Context Vectors</a>.

<img src="figure-1.png" width="500" height="350">

In case you find any of this useful, consider citing:
```
bibtex
```

## Installing Dependencies
For installing the required libraries, run the following command.
```
pip install -r requirements.txt
```

## Training

In order to begin training, run the following command.

```
python train.py
```

## Evaluation

For evaluation of the trained model using feature swapping, run the following command.
```
python style_transfer.py
```

In order to plot the latent space visualizations, run the following command.
```
python latent_visualisation.py
```
