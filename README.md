# The HIST framework for stock trend forecasting
The implementation of the paper "[HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information](https://arxiv.org/abs/2110.13716)".

## Environment
1. Install python3.7, 3.8 or 3.9. 
2. Install the requirements in [requirements.txt](https://github.com/Wentao-Xu/HIST/blob/main/requirements.txt).
3. Install the quantitative investment platform [Qlib](https://github.com/microsoft/qlib) and download the data from Qlib:
## Download the stock features of Alpha360 from Qlib
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
## Reproduce our HIST framework
```
git clone https://github.com/Wentao-Xu/HIST.git
```
# CSI 100
Set data_set='csi100' and outdir='./output/csi100_HIST', Run learn.py

# CSI 300
Set data_set='csi300' and outdir='./output/csi300_HIST', Run learn.py
