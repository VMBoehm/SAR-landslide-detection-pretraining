# SAR-landslide-detection-pretraining
Repository for the paper "SAR-based landslide classification pretraining leads to better segmentation" 

## Installing the requirements
To run the experiments presented in the paper make sure to install the requirements.

`pip install -r requirements.txt`

## Downloading the data 

Download the data from [Zenodo](https://doi.org/10.5281/zenodo.7248056). You will need the [hokkaido](https://zenodo.org/record/7248056/files/hokkaido_japan.zip) and the [kaikoura](https://zenodo.org/record/7248056/files/kaikoura_newzealand.zip) datacubes.

## Running the experiments

Follow these steps to reproduce the experiments from the paper:

1) Train models on the pretext tasks

`bash sar_landslide_pretrain/run_pretext_tasks.sh`

2) Train the downstream tasks

`bash sar_landslide_pretrain/run_segmentation_experiments.sh`

3) Analyze results and make create figures by running the notebooks in the notebook folder.

**IMPORTANT:** Before running the experiments, you will need to adapth the filepaths in the configurations files located in '/configs/experiment/'.

## Notes

The original experiments were run on an NVIDIA V100 GPU in Google Cloud.

## Citation

If you use this code for your research, please cite our paper (link will be inserted shortly).


## Acknowledgements

This work has been enabled by the Frontier Development Lab Program (FDL). FDL is a collaboration between SETI Institute and Trillium Technologies Inc., in partnership with the Department of Energy (DOE), National Aeronautics and Space Administration (NASA), the U.S. Geological Survey (USGS), Google Cloud and NVIDIA. The material is based upon work supported by NASA under award No(s) NNX14AT27A.
