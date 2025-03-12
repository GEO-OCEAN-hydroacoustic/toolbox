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