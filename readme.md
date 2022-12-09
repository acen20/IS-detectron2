The following project was built on top of the <a href="https://ai.facebook.com/tools/detectron2/">FAIR's detectron2</a>
The primary objective was to detect and segment very small objects in the image and to do that, we made some changes at the algorithm level and some at the data level.
<b>At data level</b>, we added a utility to slice the training images into 3x3 grid. This helps us retain the small objects in the image. Moreover, we get an enhanced region of interest. 
<b>At algorithm level</b>,

## Example dataset
<a href="https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7">iMaterialist (Fashion) 2020 at FGVC7 </a>
Fine-grained segmentation task for fashion and apparel

## Backbone Network
ResNet-50

### Analysis
Original Image

<img src="examples/original.jpg">

### Using different resolutions
| Low resolution (600px) | High resolution (960px) |
-------------------------|--------------------------
| <img src="examples/low_res.png"> | <img src="examples/high_res.png"> |
------------------------------------------------------------------------

### Using smaller anchor size with high resolution
| Without small anchor size | With small (16px) anchor size |
-------------------------|--------------------------
| <img src="examples/high_res_but_no_16x16_anchor.png"> | <img src="examples/high_res_with_16_anchor.png"> |


### ResNet backbone
<img src="docs/Architecture.png" />
