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

TDT, David Root and the Morales lab have created an [analysis code](https://www.tdt.com/docs/sdk/offline-data-analysis/offline-data-python/FibPhoEpocAveraging/#housekeeping) for the raw data from this fiber photometry system. <br>
I will refer to this as the Root code.
* This subtracts the isosbestic (ISOS) signal from the calcium imaging (GCaMP) signal and converts this signal to dFF and baseline Z-Score values.
* It then defines an epoch or a time window around each event (in the example below, 10 secs before and 10 secs after). The mean of these Z-score signals are then averaged across all epochs.

<p align="center">
  <img src="https://user-images.githubusercontent.com/101311642/204415712-77184a70-3ff2-46d3-a3e3-bcfc9501ebc8.png" width="400">
</p><br/>

__Purpose__

These codes create a graphical user interface (GUI) for the inputs to the Root code. <br>
They create unique event definitions, direct the Root code to perform various types of analysis and also perform post-processing.
* <ins>Types of analysis</ins>
  * Peri-events: the time around a specific event
  * Peri-events ([FED3](https://github.com/KravitzLabDevices/FED3)): the time around left nose pokes, right nose pokes and pellet drops
  * Between events: the time between events, such as 2 bottle choice, open field and elevated plus maze tests
  * Whole recording: the entire recording, and annotating any number of events over the top
* <ins>Data types</ins>
  * Notes
  * Video timestamps
  * Other epoch events
  * [EthoVision XT](https://www.noldus.com/ethovision-xt) events
* Export images and create raw data with descriptive statistics
* Create video snippets of events with the neuronal signal traces overlayed
* Create a high-throughput method for analysing many input folders with many settings

__Input data__

TDT fiber photometry systems export recording data into "[tanks](https://www.tdt.com/docs/sdk/offline-data-analysis/tdt-data-storage/)". <br>
A folder like this is the import location to the GUI. <br>
They also contain data with the structure below.

![image](https://user-images.githubusercontent.com/101311642/204426040-b153d222-57f3-46f7-884b-de7158193e6d.png)
<p align="center">
  <img src="https://user-images.githubusercontent.com/101311642/204430041-80fd069f-2f7f-492d-9cc9-cd4592801eda.png" width="600">
</p><br/>

__Output data__

Here is an example output for the peri-events (FED3) analysis. The corresponding export image is created by the Root code, like the image above.

![image](https://user-images.githubusercontent.com/101311642/204432138-ed909520-501c-4f1c-a1df-cf6ec4dba271.png)







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
