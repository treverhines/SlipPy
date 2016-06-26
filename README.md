# SlipPy
Tool for inferring static coseismic slip from GPS and InSAR data

## Features
 * Simultaneous inversion of GPS and InSAR data 
 * Laplacian smoothing
 * estimation of up to three slip components and as few as one 
 * easily customizable bounds on slip direction
 * conversions between geodetic and cartesian coordinate systems are all handled under the hood

## Limitations

 * SlipPy currently allows for only one planar fault segment
 
## Dependencies
SlipPy requires a fortran compiler for the Okada 1992 dislocation solution (thanks to Ben Thompson for making the wrapper https://github.com/tbenthompson/okada_wrapper). As for python packages, SlipPy just requires numpy, scipy, matplotlib, and basemap.   

## Installation
download SlipPy
```
$ git clone http://github.com/treverhines/SlipPy.git 
```
compile and install
```
$ cd SlipPy
$ python setup.py install
```
## Usage

A synthetic test which demonstrates all the functionality of SlipPy can be found in `example/snythetic`. This directory contains a configuration file, `config.json`, two data files, `synthetic_gps.txt`, and `synthetic_insar.txt`, and a file with the fault slip specification used to generate the synthetic data.  You can test that SlipPy is working properly by navigating to `example/synthetic` and runnning
```
$ slippy
```
SlipPy will take a few seconds to run using the default settings in `config.json`.  When finished it should have produced three files, `predicted_gps.txt`, `predicted_insar.txt`, and `predicted_slip.txt`. The format for these files is described below. For the purpose of verifying that SlipPy is working properly, I wrote a simple plotting script, `plot_slippy`, and you can view the synthetic "true" slip model and synthetic displacements with the command

```
$ plot_slippy --slip_file true_slip.txt --observed_gps_file synthetic_gps.txt
```
![alt tag](https://github.com/treverhines/SlipPy/tree/master/example/synthetic/.make_synthetic_data/true_slip.png)
