Hydroacoustic toolbox - Geo-Ocean
====================
**Contributors**: Pierre-Yves Raumer, Romain Safran
***How to use?***
1. Download the code, using for example:
   ```bash
   git clone https://github.com/GEO-OCEAN-hydroacoustic/toolbox.git
   ```



2. Install Python 3.13 (tested with 3.13.2)
   <details>
   <summary>Debian/Ubuntu</summary>
   You can run the following:
   
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update
   sudo apt-get install python3.13.2
   ```
   </details>

3. Install the required dependencies. For this, open a shell in the toolbox root directory and do:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r data/requirements.txt
   ```

4. Download the minimal & demo data (<200 MB) by running the notebook _**src/botebooks/demo/download_data.ipynb**_

5. You can test the GUI by going in the **_src_** directory with a shell and running:
   ```shell
   python -m GUI.main
   ```
   
6. You can test the different modules of the toolbox with the notebooks in _**src/notebooks/demo**_. 
   It is organized as a small tutorial that gets from reading data to the automation of event detection.

GUI
=========
**Purpose and features**

The GUI enables to visualize hydroacoustic data from .DAT, .wav or .w format.
From the main window, two options enable to load data:
- Add a sound folder: select a directory directly or indirectly containing audio data
- View an event: enable to select an event from a yaml list (parameter EVENTS_PATH in main.py) and to compute using a sound propagation the theoretical arrival time at available stations (specified in the DATASETS_CSV parameter).

Once the data from various stations are visible, the user may position the center of the spectral viewers on a given event, and use the "Locate" button to locate the source. A line will then be added in the output path (parameter OUTPUT_PATH) to save both the pick times and the location.
A right click from the user also enables to save a single pick in a file ending in "single" and located in the OUTPUT_PATH directory.


**Shortcuts and commands**
(note: a spectral view needs to be focused before using a shortcut)
- "***\+***" and "***\-***": zoom in and out
- ***Right*** and ***Left*** arrows: move on the temporal axis from half the window length
- "***\****" and "***\/***": reduce and increase max frequency in spectrogram display mode
- ***Up*** and ***Down*** arrows: move on the frequency axis
- ***Enter***: save the data of the current window in a wav file (directory of OUTPUT_PATH) and play it
- ***Ctrl + Enter***: copy the parameters of a spectral viewer (date, duration) to the other
- ***Shift + Enter***: if the parameter TISSNET_CHECKPOINT has been set, show the output of TiSSNet on the current window (same command to hide it)
- ***Left click***: move the center of the window to the position of the pointer
- ***Right click***: log the position of the pointer