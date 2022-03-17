# Unsupervised Domain Adaptation for Global Urban Extraction using Sentinel-1 and Sentinel-2 Data




Accurate and up-to-date maps of Built-Up Area (BUA) are crucial to support sustainable urban development. Earth Observation (EO) is a valuable tool to cover this demand. In particularly, the Copernicus EO program provides satellite imagery with worldwide coverage, offering new opportunities for mapping BUA at global scale. Recent urban mapping efforts achieved promising results by training Convolutional Neural Networks (CNNs) on available products using Sentinel-2 MultiSpectral Instrument (MSI) images as input, but strongly depend on the availability of local reference data for fully supervised training or assume that the application of CNNs to unseen areas (i.e.\ across-region generalization) produce satisfactory results. To alleviate these shortcomings, it is desirable to leverage Semi-Supervised Learning (SSL) algorithms which can leverage unlabeled data, especially because satellite data is plentiful. In this paper, we propose a Domain Adaptation (DA) using SSL that exploits multi-modal satellite data from Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 MSI to improve across-region generalization for BUA mapping. Specifically, two identical sub-networks are incorporated into the proposed model to perform BUA segmentation from Sentinel-1 SAR and Sentinel-2 MSI images separately. Assuming that consistent BUA segmentation should be obtained across data modality, we design an unsupervised loss for unlabeled data that penalizes inconsistent segmentation from the two sub-networks. For the final prediction, the model takes both data modalities into consideration. Experiments conducted on a test set comprised of sixty representative sites across the world demonstrate a significant improvements of the proposed DA approach (F1 score 0.703) upon fully supervised learning from Sentinel-1 SAR data (F1 score 0.581), Sentinel-2 MSI data (F1 score 0.607) and their input-level fusion (F1 score 0.653). To demonstrate the effectiveness of DA we also compared our BUA maps with a state-of-the-art product in multiple cities across the world. The comparison showed that our model produces BUA maps with comparable or even better quality.

# Our unsupervised domain adaptation approach

Overview of our unsupervised domain adaptation approach. Model parameters are optimized with a supervised loss and a consistency loss for labeled and unlabeled data, respectively. Supervised loss is comprised of three loss terms: two for the sub-networks and one for the fusion of the features extracted from the sub-networks. Consistency loss is used to optimize model parameters in a unsupervised manner by training the sub-networks to agree on their predictions. The fusion prediction is used for inference.

![](figures/domain_adaptation_workflow_revised.PNG)


# Replicating our results
## 1 Download the dataset
TBA

## 2 Train the network or download ours
TBA

## 3 Model evaluation and inference
TBA

# Credits

If you find this work useful, please consider citing:

* Sebastian Hafner, Yifang Ban, and Andrea Nascetti, "**Unsupervised Domain Adaptation for Global Urban Extraction using Sentinel-1 and Sentinel-2 Data**", *(Submitted to Remote Sensing of Environment)*, 2022

  ```bibtex

  ```
  