# MSAGA
**M**AKE **S**YSTOLIC **A**RRAY **G**REAT **A**GAIN

## Setup
This project is **_not_** a standalone project. It's designed to be integrated into chipyard framework.

To run it within chipyard:
```bash
git clone git@github.com:VCA-EPFL/chipyard-msaga.git
cd chipyard-msaga
./build-setup.sh --skip-ctags --skip-firesim --skip-marshal

```
## Unit Test
To test flash attention with a `N*N` systolic array:
```bash
cd chipyard-msaga/generators/msaga
make unit_test DIM=`N`
```
