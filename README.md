

Code for the following Paper:

Hafner, S., Ban, Y. and Nascetti, A., 2022. Unsupervised domain adaptation for global urban extraction using Sentinel-1 SAR and Sentinel-2 MSI data. Remote Sensing of Environment, 280, p.113192.

[[Paper](https://doi.org/10.1016/j.rse.2022.113192)] 

# Abstract


Accurate and up-to-date maps of Built-Up Area (BUA) are crucial to support sustainable urban development. Earth Observation (EO) is a valuable tool to cover this demand. In particularly, the Copernicus EO program provides satellite imagery with worldwide coverage, offering new opportunities for mapping BUA at global scale. Recent urban mapping efforts achieved promising results by training Convolutional Neural Networks (CNNs) on available products using Sentinel-2 MultiSpectral Instrument (MSI) images as input, but strongly depend on the availability of local reference data for fully supervised training or assume that the application of CNNs to unseen areas (i.e.\ across-region generalization) produce satisfactory results. To alleviate these shortcomings, it is desirable to leverage Semi-Supervised Learning (SSL) algorithms which can leverage unlabeled data, especially because satellite data is plentiful. In this paper, we propose a Domain Adaptation (DA) using SSL that exploits multi-modal satellite data from Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 MSI to improve across-region generalization for BUA mapping. Specifically, two identical sub-networks are incorporated into the proposed model to perform BUA segmentation from Sentinel-1 SAR and Sentinel-2 MSI images separately. Assuming that consistent BUA segmentation should be obtained across data modality, we design an unsupervised loss for unlabeled data that penalizes inconsistent segmentation from the two sub-networks. Therefore, we propose to use complementary data modalities as real-world perturbations for consistency regularization. For the final prediction, the model takes both data modalities into consideration. Experiments conducted on a test set comprised of sixty representative sites across the world demonstrate a significant improvements of the proposed DA approach (F1 score 0.694) upon fully supervised learning from Sentinel-1 SAR data (F1 score 0.574), Sentinel-2 MSI data (F1 score 0.580) and their input-level fusion (F1 score 0.651). To demonstrate the effectiveness of DA we also compared our BUA maps with a state-of-the-art product in multiple cities across the world. The comparison showed that our model produces BUA maps with comparable or even better quality.

# Our unsupervised domain adaptation approach

Overview of our unsupervised domain adaptation approach. Model parameters are optimized with a supervised loss and a consistency loss for labeled and unlabeled data, respectively. Supervised loss is comprised of three loss terms: two for the sub-networks and one for the fusion of the features extracted from the sub-networks. Consistency loss is used to optimize model parameters in a unsupervised manner by training the sub-networks to agree on their predictions. The fusion prediction is used for inference.

![](figures/domain_adaptation_workflow_revised.png)


# Replicating our results
## 1 Dataset download


The SEN12 Global Urban Mapping (SEN12_GUM) dataset can be downloaded from Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6914898.svg)](https://doi.org/10.5281/zenodo.6914898)

## 2 Network training

To train your network with our unsupervised domain adaptation approach, run the ``train_dualnetwork.py`` with the ``fusionda.yaml`` config file:

````
python train_dualnetwork.py -c fusionda -o 'path to output directory' -d 'path to GM12_GUM dataset'
````

Likewise, the baselines can be replicated by running ``train_network.py`` with the configs ``sar.yaml``, ``optical.yaml`` and ``fusion.yaml``.





## 3 Model evaluation and inference


Run the files ``testing_quantitative.py`` and ``testing_qualitative.py`` with a config of choice and the path settings from above to assess network performance. For inference, use the file ``testing_inference.py`` instead.


# Credits

If you find this work useful, please consider citing:


  ```bibtex
    @article{hafner2022unsupervised,
      title={Unsupervised domain adaptation for global urban extraction using Sentinel-1 SAR and Sentinel-2 MSI data},
      author={Hafner, Sebastian and Ban, Yifang and Nascetti, Andrea},
      journal={Remote Sensing of Environment},
      volume={280},
      pages={113192},
      year={2022},
      publisher={Elsevier}
    }
  ```
  