# Amazon molecule literature
 Literature review related to the Amazon molecule program

![](https://shields.io/badge/dependencies-Python_3.11-blue?style=flat-square)
![](https://shields.io/badge/dependencies-CUDA_11.8-green?style=flat-square)
![](https://shields.io/badge/dependencies-CuDNN_8.7.0-green?style=flat-square)

## Acknowledgements

[TCDF](https://github.com/M-Nauta/TCDF)

[alex2018](https://github.com/iancovert/Neural-GC)

## Usage

Allow different versions of CUDA and CuDNN. For settings of each method, please refer to the corresponding acknowledged repositories.

Install PyTorch version 2.1.0, corresponding to the local CUDA version.

Run the following command.

```
pip install -r requirements.txt
python method_gcastle/install.py
```

**Input:**

Copy the following files from [Amazon molecule](https://github.com/cloudy-sfu/Amazon-molecule) to this program. Users should fill the **local** file path of [Amazon molecule](https://github.com/cloudy-sfu/Amazon-molecule) program in Line 1 of the following script, and that of this program in Line 2. Do not wrap file paths with quote symbols.

```
export base_dir=
export liter_dir=
export dataset="default_15min"
cp $base_dir/raw/1_$dataset_name\_std.pkl $liter_dir/data/1_$dataset_name\_std.pkl
```

