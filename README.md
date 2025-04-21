<!-- <h1>
  <img src="docs/icon.png" alt="logon" width="200" style="vertical-align: middle; margin-right: 8px;" />
  <b style="color:red;">M</b>AKE
  <b style="color:red;">S</b>YSTOLIC
  <b style="color:red;">A</b>RRAY
  <b style="color:red;">G</b>REAT
  <b style="color:red;">A</b>GAIN !!!
</h1> -->
![](docs/msaga.png)
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
