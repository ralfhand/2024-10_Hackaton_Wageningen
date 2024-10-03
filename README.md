# 2024-10-NextGEMS-firedanger
This repository contains some pathon scripts, Jupyter Notebooks and SLURM Job examples as a starting point for the firedanger challenge at the Next-GEMS Hackaton in Wageningen, October 2024

The script contains the following files:

- prepare_input_firedanger_ICON.py
- prepare_input_firedanger_IFS.py
- prepare_input_firedanger_ICON.job
- prepare_input_firedanger_IFS.job
- compute_fwi.ipynb

These scripts extract the variables needed for the computation of the Canadian FWI, select regions, interpolate data to a regular grid and unifies the names of the relevant variables that otherwise
slightly differ between ICON and IFS. 
Customization can (and should) be done in the Customization section at the beginning of the .py scripts. 
After that, the .py scripts should not be run directly, but can be submitted as a SLURM job using the prepare_input_firedanger_ICON.job and prepare_input_firedanger_IFS.job job examples. 
By now, the example jobs are designed to run as one job per year. The year should be passed as the jobs' first argument. 
Presumably, you will usually want to process several years at onse, so you may want to submit several jobs at once. 
Therefore, e.g. use the following command to proecess all years of the ICON future projections:

for year in $(seq 2020 2050); do sbatch prepare_input_firedanger_ICON.job ${year}; done

It will take some minutes to preprocess all years, so feel free to take a short break.

During the testing of the scripts it occassionally happened that some of the output files were corrupted. To ensure that all files are ok, you could e.g. run the following command:

for file in {list of your inputfiles}; do cdo info ${file} 1>/dev/null; done

Then rerun the SLURM job for any years that are found to be damaged.

The fireweather index uses accumulative information, i.e. the current day's FWI contains information from the FWI and its subindices on the prededing days.
To account for this, your preprocessed input data should be merged into a continuous time series after the preprocessing for all years has finished. As the You can do this using cdo:

module load cdo
cdo mergetime {list of your inputfiles} {name of your outputfile}

After that you can use the python Notebook compute_fwi.ipynb to compute the actual fwi. The fireweather package usually needs installation, but during the Hackathon you can 
load a pre-installed environment from Ralf's home directory:

conda activate /home/m/m300363/.conda/envs/firedanger_dev
##### put command for kernel installation here

If you prefer to install firedanger yourself or if you even want to modify the package during the challenge, you can download it from here:

https://github.com/steidani/FireDanger

You can adapt compute_fwi.ipynb to make plots or export the FWI or subcomponents of it to a data file.

More information on the Canadian Fire Weather index can be found e.g. here:

https://climatedataguide.ucar.edu/climate-data/canadian-forest-fire-weather-index-fwi
