## Instance Segmentation
Suitable for small objects, trains on additional anchor boxes for Instance segmentation

## Dataset
iMaterial Fashion FGVC7

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
