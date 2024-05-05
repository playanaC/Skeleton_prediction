# Automatic segmentation of *Caenorhabditis elegans* skeletons in worm aggregations using improved U-Net in low-resolution image sequences.
![image](https://github.com/playanaC/Skeleton_prediction/blob/main/pipeline_graph.png)

Pose estimation of *C. elegans* in image sequences is challenging and even more difficult in low-resolution images. Problems range from occlusions, loss of worm identity, and overlaps to aggregations that are too complex or difficult to resolve, even for the human eye. Neural networks, on the other hand, have shown good results in both low-resolution and high-resolution images. However, training in a neural network model requires a very large and balanced dataset, which is sometimes impossible or too expensive to obtain.
In this article, a novel method for predicting *C. elegans* poses in cases of multi-worm aggregation and aggregation with noise is proposed. To solve this problem we use an improved U-Net model capable of obtaining images of the next aggregated worm posture. This neural network model was trained/validated using a custom-generated dataset with a synthetic image simulator. Subsequently, tested with a dataset of real images. The results obtained were greater than 75% in precision and 0.65 with Intersection over Union (IoU) values.

# Proposed neural network model
![image](https://github.com/playanaC/Skeleton_prediction/blob/main/model.PNG)


# Tracking algorithm
![image](https://github.com/playanaC/Skeleton_prediction/blob/main/tracking_algoritm.png)


# Dataset:
- The real dataset with all experiments can be downloaded from [dataset_skeletons](https://active-vision.ai2.upv.es/wp-content/uploads/2021/02/dataset_skeletons.zip).


# Image adquisition system:
- Images were captured by an [open hardware system](https://github.com/JCPuchalt/SiViS).


# Examples for aggregation and noise
![image](https://github.com/playanaC/Skeleton_prediction/blob/main/example_results.png)


# References:
- Puchalt, J. C., Sánchez-Salmerón, A.-J., Martorell Guerola, P. & Genovás Martínez, S. "Active backlight for automating visual monitoring: An analysis of a lighting control technique for *Caenorhabditis elegans* cultured on standard Petri plates". PLOS ONE 14.4 (2019) [doi paper](https://doi.org/10.1371/journal.pone.0215548)

- Puchalt, J.C., Sánchez-Salmerón, A.-J., Ivorra, E. "Improving lifespan automation for *Caenorhabditis elegans* by using image processing and a post-processing adaptive data filter". Scientific Reports (2020) [doi paper](https://doi.org/10.1038/s41598-020-65619-4).

- Layana Castro Pablo E., Puchalt, J.C., Sánchez-Salmerón, A.-J. "Improving skeleton algorithm for helping *Caenorhabditis elegans* trackers". Scientific Reports (2020) [doi paper](https://doi.org/10.1038/s41598-020-79430-8).

- Layana Castro Pablo E., Puchalt, J.C., García Garví, A., Sánchez-Salmerón, A.-J. "*Caenorhabditis elegans* Multi-Tracker Based on a Modified Skeleton Algorithm". Sensors (2021) [doi paper](https://doi.org/10.3390/s21165622).

- Puchalt, J.C., Sánchez-Salmerón, A.-J., Ivorra, E., Llopis, S., Martínez, R., Martorell, P. "Small flexible automated system for monitoring *Caenorhabditis elegans* lifespan based on active vision and image processing techniques.". Scientific Reports (2021) [doi paper](https://doi.org/10.1038/s41598-021-91898-6).

  # Citation:
```
@article{Layana2023H,
  title={Automatic segmentation of *Caenorhabditis elegans* skeletons in worm aggregations using improved U-Net in low-resolution image sequences},
  author={Castro, Pablo E Layana and Garví, Antonio García and Sánchez-Salmerón, Antonio-José},
  journal={Heliyon},
  volume={9},
  number={4},
  year={2023},
  publisher={Elsevier}
}
```
