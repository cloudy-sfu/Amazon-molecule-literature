# Amazon molecule literature
 Literature review related to the Amazon molecule program

![](https://shields.io/badge/dependencies-Python_3.11-blue?style=flat-square)
![](https://shields.io/badge/dependencies-CUDA_11.8-green?style=flat-square)
![](https://shields.io/badge/dependencies-CuDNN_8.7.0-green?style=flat-square)
![](https://shields.io/badge/OS-Ubuntu_20.02-lightgrey?style=flat-square)

## Acknowledgments

[TCDF](https://github.com/M-Nauta/TCDF)

[Alex2018](https://github.com/iancovert/Neural-GC)

[eSRU](https://github.com/sakhanna/SRU_for_GCI)

[gcastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)

[GVAR](https://github.com/i6092467/GVAR)

## Usage

1. Find the following release version of PyTorch `torch-2.1.0+cu118-cp311-cp311-linux_x86_64.whl`
2. Use  `pip install torch-2.1.0+cu118-cp311-cp311-linux_x86_64.whl ` to install PyTorch.
3. Run the following command.
    ```
    pip install -r requirements.txt
    python file_exchange.py
    python method_gcastle/install.py
    ```

