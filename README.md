# SAR-landslide-detection-pretraining
Repository for the paper "SAR-based landslide classification pretraining leads to better segmentation" (accepted at [AI+HADR](https://www.hadr.ai/home) @NeurIPS 2022)

## Installing the requirements
To run the experiments presented in the paper make sure to install the requirements.

`pip install -r requirements.txt`

## Downloading the data 

Download the data from [Zenodo](https://doi.org/10.5281/zenodo.7248056). You will need the [hokkaido](https://zenodo.org/record/7248056/files/hokkaido_japan.zip) and the [kaikoura](https://zenodo.org/record/7248056/files/kaikoura_newzealand.zip) datacubes.

## Running the experiments

Follow these steps to reproduce the experiments from the paper:

1) Train models on the pretext tasks

`bash ./scripts/run_pretext_tasks.sh`

2) Train the downstream tasks

`bash ./scripts/run_segmentation_experiments.sh`

3) Analyze results and create figures by running the notebooks in the [notebook folder](https://github.com/VMBoehm/SAR-landslide-detection-pretraining/tree/main/notebooks).

**IMPORTANT:** Before running the experiments, you will need to adapt the filepaths in the configurations files located in [configs/experiment/](https://github.com/VMBoehm/SAR-landslide-detection-pretraining/tree/main/configs/experiment).

## Notes

The original experiments were run on an NVIDIA V100 GPU in Google Cloud.

## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2211.09927):

```
@misc{https://doi.org/10.48550/arxiv.2211.09927,
  doi = {10.48550/ARXIV.2211.09927},
  
  url = {https://arxiv.org/abs/2211.09927},
  
  author = {Böhm, Vanessa and Leong, Wei Ji and Mahesh, Ragini Bal and Prapas, Ioannis and Nemni, Edoardo and Kalaitzis, Freddie and Ganju, Siddha and Ramos-Pollan, Raul},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Image and Video Processing (eess.IV), Signal Processing (eess.SP), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  
  title = {SAR-based landslide classification pretraining leads to better segmentation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```


## Acknowledgements

This work has been enabled by the Frontier Development Lab Program (FDL). FDL is a collaboration between SETI Institute and Trillium Technologies Inc., in partnership with the Department of Energy (DOE), National Aeronautics and Space Administration (NASA), the U.S. Geological Survey (USGS), Google Cloud and NVIDIA. The material is based upon work supported by NASA under award No(s) NNX14AT27A.
