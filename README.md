# Scarlet-Lycoris
## RF Fingerprint Analysis Based on RTL-SDR and Random Forest
### How to use
Use [rtl_power.exe](https://osmocom.org/projects/rtl-sdr/wiki) to create a CSV file.If you want to use a model in this project,you should set the gain to 40 dB, step 2 kHz, and 2 MHz span,Example:
```bash
rtl_power.exe -f 437M:439M:2k -g 40 -i 1s -e 10s scan.csv
```
Then
```bash
python main.py scan.csv
```
### How to train
Standardize the CSV file(refer to main.py),and put them into /samplgite and run train.py
#### This project is based on spurious emissions so it is non-practical and just for fun