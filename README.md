# Quickstart: Use the TWS API with Python
## Overview
This repository contains a Python script that demonstrates how to use the TWS API to connect to a running TWS or IB Gateway instance and place a market order.
## Requirements
* Python 3.6 or later
* The TWS API
* A running TWS or IB Gateway instance
## Usage
1. Clone this repository to your local machine.
2. Change your shell's current directory to `quantum_t`.
3. Install the required packages by running `pip install -r requirements.txt` in your terminal.
4. `[options]`Open `DataCollection/config.py`,you can change options in the file.
5. Run the script by running in your terminal:

windows:
```python
python .\quantum_t\DataCollection\QQQ3D.py
```
linux:
```python
python ./quantum_t/DataCollection/QQQ3D.py
```
Also you can run as this in windows:
```python
python .\DataCollection\QQQ3D.py --bar_size='1 min' --num_days=20 --contract_symbol=NDX
```
the csv file will be saved as `NDX`_dataset.csv