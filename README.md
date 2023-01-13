# intelli-recon

IntelliRecon uses deep learning to monitor and detect changes in satellite images. It also classifies illegal military
activities in the country's border region. Part of the ASDS21 Computer Vision course final exam. “IntelliRecon" reflects
the intelligent and advanced nature of the system, as well as its focus on reconnaissance and surveillance.

# Usage

## Data 
You can find dataset in this [link](https://www.kaggle.com/code/aninda/change-detection-nb/data). 
For Data keep this structure. 
```shell
--data
----images
---- # folders with city names etc. 
----train_labels
---- # folders with city names etc.
```

## Installation

```shell
conda creante -n intelli-recon python=3.8.16

pip install git+https://github.com/mherkhachatryan/intelli-recon.git
```

## Configuration

Configs can be changed from `recon/configs.py`.
You just need to change `# path setting` section, all else is for experiments.

## Run

From repo level (`intelli-recon/` ) run

```shell
python recon/
```

# Model Weights
Best model's configurations and weights can be found at this [link](https://app.neptune.ai/mherkhachatryan/intelli-recon/experiments?split=tbl&dash=charts&viewId=983493ff-7039-4975-a552-be0e058b84fd)
with tag `final`. You can track other experiments here too. 

To test model, in `config.py` pass  `MODE` to `valid` or `test`. 