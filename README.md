# TDT (Tucker-Davis Technologies) Fiber Photometry Analysis GUI 🐁

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

Here is an example output for the peri-events (FED3) analysis. The corresponding export image is created by the Root code, like the image above. The peri-events and between events analysis types also have the same output format.

![image](https://user-images.githubusercontent.com/101311642/204432138-ed909520-501c-4f1c-a1df-cf6ec4dba271.png)

Here is an example export image for the whole recording analysis, with the notes and EthoVision events annotated.

![A_AgrpCre2837-220222-122913_zScore_Whole recording_Setup_A](https://user-images.githubusercontent.com/101311642/204433171-7fc272b2-c214-465e-bd38-b24afde35532.png)

Here is an example video snippet of the air puff event with the Z-Score signal trace below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/101311642/182095128-09a8b226-88e7-4976-ac26-e17be3b6123a.gif"/ width="60%">
</p><br/>

__High throughput analysis of TDT tanks__

After every analysis run using the GUI, a settings excel file is exported. These settings can be edited and also duplicated. When this file is re-imported, it can run the analysis for multiple tanks at once. If you set "import subfolders" to TRUE, you can also apply the same analysis to all tanks within one master folder.

![image](https://user-images.githubusercontent.com/101311642/204436281-95b738aa-94b2-4bd8-b08b-31de052ccda6.png)

__Preview of the graphical user interfaces__

![image](https://user-images.githubusercontent.com/101311642/204446219-559732dc-19d0-4000-948a-71f717e5d236.png)

### Installation

Install [Anaconda Navigator](https://www.anaconda.com/products/distribution). <br>
Open Anaconda Prompt (on Mac open terminal and install X-Code when prompted). <br>
Download this repository to your home directory by typing in the line below.
```
git clone https://github.com/Andrews-Lab/Fiber_photometry_analysis.git
```
If you receive an error about git, install git using the line below, type "Y" when prompted and then re-run the line above.
```
conda install -c anaconda git
```
Change the directory to the place where the downloaded folder is. <br>
```
cd Fiber_photometry_analysis
```

Create a conda environment and install the dependencies.
```
conda env create -n FPA -f Dependencies.yaml
```

### Usage
Open Anaconda Prompt (on Mac open terminal). <br>
Change the directory to the place where the git clone was made.
```
cd Fiber_photometry_analysis
```

Activate the conda environment.
```
conda activate FPA
```

Run the codes.
```
python Run_program.py
```

### Guide

Read the [guide to using this fiber photometry GUI](How_to_use_Fiber_Photometry_GUI.pdf)

<br>

### Acknowledgements

__Author:__ <br>
[Harry Dempsey](https://github.com/H-Dempsey) (Andrews lab and Foldi lab) <br>

__Credits:__ <br>
Tucker-Davis Technologies, David Root, Morales lab <br>
Alexander Reichenbach, Zane Andrews, Laura Milton, Kyna Conn, Claire Foldi <br>

__About the labs:__ <br>
The [Root lab](https://www.root-lab.org/) utilizes molecular, electrophysiological, imaging, and viral-based approaches to define, observe, and manipulate the specific neurons and pathways that govern seeking rewards, avoiding harm, addiction, and stress. <br>
The [Morales lab](https://irp.drugabuse.gov/staff-members/marisela-morales/) investigates the molecules, cells and neuronal pathways central to the neurobiology of drug addiction. <br>
The [Andrews lab](https://www.monash.edu/discovery-institute/andrews-lab) investigates how the brain senses and responds to hunger. <br>
The [Foldi lab](https://www.monash.edu/discovery-institute/foldi-lab) investigates the biological underpinnings of anorexia nervosa and feeding disorders. <br>
