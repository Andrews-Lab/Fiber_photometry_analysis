# TDT (Tucker-Davis Technologies) Fiber Photometry analysis GUI 🐁

### Overview

__TDT Fiber Photometry__

The [TDT Fiber Photometry System](https://www.tdt.com/system/fiber-photometry-system/) measures the activity of neurons in rodents using calcium imaging.
These real-time neuronal activity measurements can also be [coupled with behavioural events](https://www.tdt.com/system/behavior-and-fiber-photometry/).
For example, this could be a rodent poking a Feeding Experimentation Device ([FED3](https://github.com/KravitzLabDevices/FED3)), 
entering the closed arm of an elevated plus maze or reacting to the drop of a peanut butter chip.

<p align="center">
  <img src="https://user-images.githubusercontent.com/101311642/204413263-5d4cb7f5-b4be-4d7d-a80b-29ca202ff596.png" width="400">
</p><br/>

TDT, David Root and the Morales lab have created an [analysis code](https://www.tdt.com/docs/sdk/offline-data-analysis/offline-data-python/FibPhoEpocAveraging/#housekeeping) for the raw data from this fiber photometry system.
* This subtracts the isosbestic (ISOS) signal from the calcium imaging (GCaMP) signal and converts this signal to dFF and baseline Z-Score values.

<p align="center">
  <img src="https://user-images.githubusercontent.com/101311642/204415712-77184a70-3ff2-46d3-a3e3-bcfc9501ebc8.png" width="400">
</p><br/>

__Purpose__

The CSV output from the FED3 devices show the timestamps of each event, like nose pokes or pellet retrievals. This repository :
* Converts this output into a time binned file. It also adds another sheet with the time stamps of all pellet count changes.
* Creates a master file that combines all the “Left poke count” columns from the raw FED files into one sheet. It does the same thing for the other column types as well. The columns are then sorted by genotype and treatment. <br>

__Preview of the graphical user interfaces__

![image](https://user-images.githubusercontent.com/101311642/195033127-046fec78-24ae-4ab7-b059-f763a19e93b4.png)

__Input and output data__

![image](https://user-images.githubusercontent.com/101311642/194794376-e8ae77ac-dbc8-41dc-a1c8-bf0b7ace3f52.png)

### Installation

Install [Anaconda Navigator](https://www.anaconda.com/products/distribution). <br>
Open Anaconda Prompt (on Mac open terminal and install X-Code when prompted). <br>
Download this repository to your home directory by typing in the line below.
```
git clone https://github.com/Andrews-Lab/FED3_time_bins.git
```
If you receive an error about git, install git using the line below, type "Y" when prompted and then re-run the line above.
```
conda install -c anaconda git
```
Change the directory to the place where the downloaded folder is. <br>
```
cd FED3_time_bins
```

Create a conda environment and install the dependencies.
```
conda env create -n FTB -f Dependencies.yaml
```

### Usage
Open Anaconda Prompt (on Mac open terminal). <br>
Change the directory to the place where the git clone was made.
```
cd FED3_time_bins
```

Activate the conda environment.
```
conda activate FTB
```

Run the codes.
```
python FED.py
```

### Guide

View the guide about [how to analyse your FED data](How_to_use_FED_code.pdf).

<br>

### Acknowledgements

__Author:__ <br>
[Harry Dempsey](https://github.com/H-Dempsey) (Andrews lab and Foldi lab) <br>

__Credits:__ <br>
Zane Andrews, Wang Lok So, Lex Kravitz <br>

__About the labs:__ <br>
The [Andrews lab](https://www.monash.edu/discovery-institute/andrews-lab) investigates how the brain senses and responds to hunger. <br>
The [Foldi lab](https://www.monash.edu/discovery-institute/foldi-lab) investigates the biological underpinnings of anorexia nervosa and feeding disorders. <br>
The [Kravitz lab](https://kravitzlab.com/) investigates the function of basal ganglia circuits and how they change in diseases such as obesity, addiction, and depression. <br>
