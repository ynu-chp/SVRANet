## Introduction

This is the Pytorch implementation of our paper:`Transformer-Based Mechanism Design for Semantic Communications IoV Crowdsensing Service` .


## Requirements


* Python >= 3.7
* Pytorch 1.10.0
* Argparse
* Logging
* Tqdm
* Scipy

## Usage

### Generate the data

```bash
python generate_data.py
#You can set the number of users
```

### Train SVRANet

```bash
#es=3
# bidder=2
python 2x3.py

# bidder=3
python 3x3.py

# bidder=4
python 4x3.py

# bidder=5
python 5x3.py

# bidder=6
python 6x3.py

# bidder=7
python 7x3.py

```

## Acknowledgement

Our code is built upon the implementation of <https://arxiv.org/abs/2201.12489>.

