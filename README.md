# Chest X-Ray Images (Pneumonia)

## Context

[From Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


![Figure_1](./doc/resources/figure_1.png)

Figure 1. Illustrative Examples of Chest X-Rays in Patients with Pneumonia

The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.

[Original article](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

## Content

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Acknowledgements

Data: https://data.mendeley.com/datasets/rscbjbr9sj/2

License: CC BY 4.0

Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

## START

### Create Environment

To execute the notebooks we need to create the environment

```shell
conda env create -f environment.yml
```

### Create Kaggle API Token

If you don't have a API Token from kaggle you need to create one to follow the steps or you can download manually the data and set in the local location

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```
