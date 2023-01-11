# intelli-recon

IntelliRecon uses deep learning to monitor and detect changes in satellite images. It also classifies illegal military
activities in the country's border region. Part of the ASDS21 Computer Vision course final exam. “IntelliRecon" reflects
the intelligent and advanced nature of the system, as well as its focus on reconnaissance and surveillance.

# Usage

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
