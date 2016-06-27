# SlipPy
SlipPy (pronounced slip-ee) is a tool for inferring static coseismic slip from GPS and InSAR data

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

A synthetic test which demonstrates all the functionality of SlipPy can be found in `example/snythetic`. This directory contains a configuration file, `config.json`, two data files, `synthetic_gps.txt`, and `synthetic_insar.txt`, and a file with the fault slip specification used to generate the synthetic data.  You can test that SlipPy is working properly by navigating to `example/synthetic` and running
```
$ slippy
```
SlipPy will take a few seconds to run using the default settings in `config.json`.  When finished it should have produced three files, `predicted_gps.txt`, `predicted_insar.txt`, and `predicted_slip.txt`. The format for these files is described below. For the purpose of verifying that SlipPy is working properly, I wrote a simple plotting script, `plot_slippy`, and you can view the synthetic "true" slip model and synthetic displacements with the command
```
$ plot_slippy --slip_file true_slip.txt --observed_gps_file synthetic_gps.txt
```
To view the predicted slip model and compare the synthetic and predicted gps displacements  
```
$ plot_slippy --slip_file predicted_slip.txt --observed_gps_file synthetic_gps.txt \
--predicted_gps_file predicted_gps.txt
```
Although the synthetic inversion uses synthetic InSAR data. I currently do not have a good way of plotting the InSAR data.

### configuration file
All of the necessary user defined parameters are given in the file `config.json`, which contains the following
```
{
"basis1":[1.0,1.0,0.0],
"basis2":[1.0,-1.0,0.0],
"length":200000,
"width":60000,
"Nlength":30,
"Nwidth":10,
"penalty":50.0,
"strike":70.0,
"dip":45.0,
"position":[-84.2,43.3,0.0],
"gps_input_file":"synthetic_gps.txt",
"insar_input_file":"synthetic_insar.txt",
"gps_output_file":"predicted_gps.txt",
"insar_output_file":"predicted_insar.txt",
"slip_output_file":"predicted_slip.txt"
}
```
You can view a complete description of all of these parameters with the command, 
```
$slippy -h   
```
SlipPy uses the values in `config.json` as default values, which can then be overwritten with a command line argument. For example, to invert for slip on a fault with a 40 degree dip you can use the command
```
slippy --dip 40
```
If a necessary parameter is not given in either the config.json file or via the command line then slippy will inform you of the missing parameter.

## Data files
The input and output files for GPS data have the same format.  They contain a header line, which gets ignored, and 8 columns of space separated data.  Here are the first few rows of the synthetic GPS file, which should be self explanatory 
```
# lon[degrees] lat[degrees] disp_e[m] disp_n[m] disp_v[m] sigma_e[m] sigma_n[m] sigma_u[m]
-83.8991 42.6473 0.0576 0.0219 0.0007 0.0000 0.0000 0.0000
-83.4448 44.3117 -0.0080 -0.0166 0.0041 -0.0000 -0.0000 0.0000
-84.1151 43.5458 -0.0449 -0.0257 0.0018 -0.0000 -0.0000 0.0000
-84.5774 43.8300 -0.0209 -0.0047 -0.0007 -0.0000 -0.0000 -0.0000
-83.7757 43.6817 -0.0313 -0.0321 0.0051 -0.0000 -0.0000 0.0000
...
```
The input and output files for insar data also have the same format. Here are the first few lines for the synthetic insar data file
```
# lon[degrees] lat[degrees] disp_los[m] sigma_los[m] V_e V_n V_u
-83.8991 42.6473 0.0757 0.0100 0.5773 0.5773 0.5773
-83.4448 44.3117 -0.0228 0.0100 0.5773 0.5773 0.5773
-84.1151 43.5458 -0.0648 0.0100 0.5773 0.5773 0.5773
-84.5774 43.8300 -0.0222 0.0100 0.5773 0.5773 0.5773
...
```
disp_los, sigma_lost are the displacements and uncertainties along the look direction. The look direction, which is the vector pointing from the observation point to the satellite, is given by Ve, V_n, and V_u. The line of sight vector should be normalized to 1.0.      

SlipPy produces an output file containing the geometric and slip information for each fault patch.  Here is an example

```
# lon[degrees] lat[degrees] depth[m] strike[degrees] dip[degrees] length[m] width[m] left-lateral[m] thrust[m] tensile[m]
-85.1514 42.6431 -41012.1933 70.0000 45.0000 3333.3333 2000.0000 0.0952 0.0000 0.0000
-85.1576 42.6550 -39597.9797 70.0000 45.0000 3333.3333 2000.0000 0.1012 0.0000 0.0000
-85.1637 42.6669 -38183.7662 70.0000 45.0000 3333.3333 2000.0000 0.1074 0.0000 0.0000
-85.1699 42.6788 -36769.5526 70.0000 45.0000 3333.3333 2000.0000 0.1140 0.0000 0.0000
...
```
lon, lat, and depth describe the location of the top center of each fault patch (the subsurface always has a negative depth).  The left-lateral, thrust, and tensile components of slip are alway returned regardless the choice of basis vector used as input. Strike is given using the right-hand rule, where the fault is dipping to the right when facing in the direction of strike.

### Slip basis vectors
The slip basis vectors, which are specified with the arguments 'basis1', 'basis2', or 'basis3', are used to bound the slip solution.  Slip solutions are constrained to be within the positive span of the slip basis vectors.  This is perhaps best illustrated with an example.  Suppose our config.json file has no basis entried in it.  We can invert for slip with the constraint that slip is left-lateral with the command
```
slippy --basis1 1.0 0.0 0.0
```
where the arguments are for the left-lateral, thrust, and tensile component of our only slip basis vector. If we want slip to be within 45 degrees of left lateral then we need to supply two slip basis functions with the command
```
slippy --basis1 1.0 1.0 0.0 --basis2 1.0 -1.0 0.0
```
there is no need to normalize the slip basis components.










