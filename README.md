# Biodeg
Predict if a molecule is generally degraded when in the wild.

### Install
#### From source
```bash
conda create -n biodeg python=3.11 && \
    conda activate biodeg && \
    pip install .
```
#### From github
Download a release from [releases](https://github.com/research11111/biodeg/releases), then
```bash
conda create -n biodeg python=3.11 && \
    conda activate biodeg && \
    pip install biodeg-*.whl
```
### Developp
#### Install
```bash
conda create -n biodeg python=3.11 && \
    conda activate biodeg && \
    pip install -e . && \
    pip install poetry
```
#### Run
```bash
biodeg train -e 10 -i data/data/All-Public_dataset_Mordred_tail_10.csv -o /tmp/e.pt
```
#### Build
```bash
poetry build
```
#### Test
```bash
poetry run pytest
```
