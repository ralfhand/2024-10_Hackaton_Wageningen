# 2024-10-NextGEMS - Firedanger challenge software package

This repository contains some pathon scripts, Jupyter Notebooks and SLURM Job examples as a starting point for the firedanger challenge at the Next-GEMS Hackaton in Wageningen, October 2024

The repo contains the following files:

- prepare_input_firedanger_ICON.py
- prepare_input_firedanger_IFS.py
- prepare_input_firedanger_ICON.job
- prepare_input_firedanger_IFS.job
- compute_fwi.ipynb
- plot_input_and_fwi_components.ipynb

you can clone this repository by using the following command in your levante home directory:

```
mkdir [directory name of your choice] && cd [directory name of your choice]
git clone https://gitlab.dkrz.de/m300363/2024-10_hackaton_wageningen .

```

## Preparation of fire danger variables using prepare_input_firedanger_*

These scripts named **prepare_input_firedanger_ICON.py** and **prepare_input_firedanger_ICON.py** extract the variables needed for the 
computation of the Canadian Fireweather index, select regions, interpolate data to a regular grid and unifies the names of the relevant variables
that otherwise slightly differ between ICON and IFS. Customization can (and should) be done in the Customization section at the beginning of the .py scripts.

After that, the .py scripts should not be run directly, but can be submitted as a SLURM job using the ***prepare_input_firedanger_ICON.job***
and ***prepare_input_firedanger_IFS.job*** job examples. 

By now, the jobs were optimized and tested for/on a 10°x10° region (California). Enlarging the region may require changes in the SLURM settings of
the job scripts, particularly concerning the mem/tasks-per-node parameters. I recommend to do so only if you are familiar to SLURM. 
Please avoid processing large regions and requesting excessive resources during the Hackathon, as otherwise your jobs might stuck in the job queue for
a very long time given the fact that many people will work simultanously on levante during this week. Please also be responsible concerning the use
of disk space in the project folder.

By now, the example jobs are designed to run as one job per year. The year should be passed as the jobs' first argument. 
Presumably, you will usually want to process several years at once, so you may want to submit several jobs at once. 
Therefore, e.g. use the following command to proecess all years of the ICON future projections:

```
sbatch --array=2020-2049 prepare_input_firedanger_ICON.job
```

For ICON it will take some minutes to preprocess all years, so feel free to take a short break.
For IFS processing takes longer. Probably the jobs need to run overnight. 

During the testing of the scripts it occassionally happened that some of the output files were corrupted. To ensure that all files are ok, you could e.g. run the following command:

```
for file in {list of your inputfiles}; do cdo info ${file} 1>/dev/null; done
```

Then rerun the SLURM job for any years that are found to be damaged.

The fireweather index uses accumulative information, i.e. the current day's FWI contains information from the FWI and its subindices on the preceding days.
To account for this, your preprocessed input data should be merged into a continuous time series after the preprocessing for all years has finished. As the You can do this using cdo:

```
module load cdo
cdo mergetime {list of your inputfiles} {name of your outputfile}
```

## Computing the Canadian FWI using compute_fwi.ipynb and exploring it with plot_input_and_fwi_components.ipynb

After that you can use the python Notebook **compute_fwi.ipynb** to compute the Canadian Fire Weather index. The script uses a python package that was written at the University of Bern by Daniel Steinfeld. The package usually needs installation, but during the Hackathon you can load a pre-installed environment from Ralf's home directory:

```
# activate firedanger environment
conda activate /home/m/m300363/.conda/envs/firedanger_dev

# install kernel in environment
python -m ipykernel install --user --name=firedanger_dev
```

If you prefer to install firedanger yourself or if you even want to modify the package during the challenge, you can download it from here:

https://github.com/steidani/FireDanger

You can then use the compute_fwi.ipynb notebook to compute your FWIs. Therefore please adapt (at least) the file names and paths in the notebook.

For a quick inspection of the input data and your computed FWIs you can adapt the **plot_input_and_fwi_components.ipynb** notebook

More information on the Canadian Fire Weather index can be found e.g. here:

https://climatedataguide.ucar.edu/climate-data/canadian-forest-fire-weather-index-fwi

## Precomputed input data on Levante

Pre-computed input data and FWIs computed during the testing of these scripts can be found here:

levante:/work/bb1153/m300363/fireweather_data/California

# Happy Hacking!!!



