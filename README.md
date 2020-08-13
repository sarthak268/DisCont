# DisCont: Self-Supervised Visual Attribute Disentanglement using Context Vectors

This repository contains code for the paper <a href="https://arxiv.org/abs/2006.05895"><i>DisCont</i>: Self-Supervised Visual Attribute Disentanglement using Context Vectors</a>. Video available <a href="https://youtu.be/2gyLEPnTi1M">here</a>.

## Abstract
Disentangling the underlying feature attributes within an image with no prior supervision is a challenging task. Models that can disentangle attributes well provide greater interpretability and control. In this paper, we propose a self-supervised framework <i>DisCont</i> to disentangle multiple attributes by exploiting the structural inductive biases within images. Motivated by the recent surge in contrastive learning paradigms, our model bridges the gap between self-supervised contrastive learning algorithms and unsupervised disentanglement. We evaluate the efficacy of our approach, both qualitatively and quantitatively, on four benchmark datasets.

![DisCont Training](figure.jpg)

In case you find any of this useful, consider citing:
```
@article{Bhagat2020DisContSV,
  title={DisCont: Self-Supervised Visual Attribute Disentanglement using Context Vectors},
  author={Sarthak Bhagat and Vishaal Udandarao and Shagun Uppal},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.05895}
}
```

## Installing Dependencies
In order to install the required libraries, clone our repository and run the following command:
```
pip install -r requirements.txt
```

## Preparing Data
In our paper, we evaluate the efficacy of our approach on a set of four publicly available datasets. Download any of these datasets and place them inside another folder in order to begin training.
* <a href="http://www-personal.umich.edu/~reedscot/files/nips2015-analogy-data.tar.gz">Sprites</a>
* <a href="http://www-personal.umich.edu/~reedscot/files/nips2015-analogy-data.tar.gz">Cars3D</a>
* <a href="https://github.com/deepmind/3d-shapes">3DShapes</a>
* <a href="https://github.com/deepmind/dsprites-dataset/">dSprites</a>

## Training

Begin training the <i>DisCont</i> model by running the following script.
```
python train.py
```

Customize training by varying the latent structure using the given set of flags.
```
--z_chunk_size             Dimension of each Latent Chunk
--z_num_chunks             Number of Latent Chunks
--c_chunk_size             Dimension of each Context Vector Chunk
--c_num_chunks             Number of Context Vector Chunks
--num_specified_chunks     Number of Specified Chunks in the Latent Space
--num_unspecified_chunks   Number of Unspecified Chunks in the Latent Space
```

## Evaluation

For evaluation of the trained model using feature swapping, run the following command.
```
python style_transfer.py
```

In order to plot the latent space visualizations, run the following command.
```
python latent_visualization.py
```

## Contact
If you face any problem in running this code, you can contact us at {sarthak16189, vishaal16119, shagun16088}@iiitd.ac.in.

## License
Copyright (c) 2020 Sarthak Bhagat, Vishaal Udandarao, Shagun Uppal.

For license information, see LICENSE or http://mit-license.org
