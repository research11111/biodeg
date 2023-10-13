# Biodeg
Predict if a molecule is generally degraded when in the wild.

### Install
#### From source
```bash
conda create -n biodeg python=3.11 && \
    conda activate biodeg && \
    pip install .
```
### Developp
#### Install
```bash
conda create -n biodeg python=3.11 && \
    conda activate biodeg && \
    pip install -e . && \
    pip install poetry
```
#### Build
```bash
poetry build
```
