# Data

Datasets are not stored in this repo. Download them from the sources below.

## Stream A: histopathology patches
Kaggle: **sunilthite/ovarian-cancer-classification-dataset**
(https://www.kaggle.com/datasets/sunilthite/ovarian-cancer-classification-dataset)

Notebook 01 downloads it automatically once you set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
The archive unpacks into `Train_Images/<SUBTYPE>/` and `Test_Images/<SUBTYPE>/`.

## Stream B: clinical blood panel
File: **`Supplementary data 1.xlsx`**
- 349 patients and 49 features.
- The `TYPE` column is the label: 1 = ovarian cancer, 0 = benign ovarian tumour.
- The file is public and is also mirrored on Kaggle as "Predict Ovarian Cancer".

To run the demo app, place the file here as `data/Supplementary data 1.xlsx`.
