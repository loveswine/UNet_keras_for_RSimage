# U-Net Semantic Segmentation For Remote Sensing Images

| ![](./docs/image.png) | ![](./docs/label.png) | ![](./docs/predict.png) |
| :--------------------: | :-------------------: | :---------------------: |
|         image          |         label         |         predict         |


![](./docs/model.png)


## Installation

### Requirements

* Windows, Linux

* Python 3.6+ 

* Keras=2.31

* tensorflow=1.14

* CUDA 9.0 or higher

* GDAL  ``` pip install ./package/GDAL-3.1.4-cp36-cp36m-win_amd64.whl```

  

## Dataset

Here an example is given by using [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/). and

[Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery | Zenodo](https://zenodo.org/record/5171712)

### Data descriptions

- /train/ - this folder contains the training set images

  /train/ image/ - the folder contains the images who have been cut to a specific size from remote sensing images 

  /train/ label/ - the folder contains the labels corresponding to the /train/ image/ 

- /val/ - this folder contains the validation set images consistent with the training set structure

- /test/ - this folder contains the test set images consistent with the training set structure

  

How to use it?
---------------------

Directly run **train.py** functions with different network parameter settings to produce the results. 

**test.py** can predict images in test set and save them, after that **iou.py** can calculate oa, F1 score Etc. on test set

**predict_rsimage.py** can predict a single large Remote sensing image and save it

**split.py** can split Remote sensing images to specific size for building dataset structure





##  Acknowledgements

I have used utility functions from other wonderful open-source projects. Espeicially thank the authors of:

https://github.com/YanjieZe/UNet

https://zhuanlan.zhihu.com/p/158769096

https://zhuanlan.zhihu.com/p/163682002