Setup:
This folder contains OPF results generated under varying load scenarios with Gaussian noise.
By varying P_demand, Q_demand, and topology, the optimization process determines the optimal P_gen, Q_gen, and total generation cost.

--------------------------------------------------
File Introduction
--------------------------------------------------

Each output file is named as:
    results_<case>_<topology>_sigma<noise_std>.csv

Example:
    results_case30_topo0_sigma0.01.csv

- <topology>: The power system case used (e.g., case30_topo0, case30_topo1, etc.).
- <noise_std>: The std (\sigma) of the Gaussian noise applied to active and reactive power loads (fixed power factor).

Note: 
Same <noise_std> implies identical Pd and Qd values across topologies due to fixed random seed. 

Note: 
topo0 -> topo1: delete one line
topo1 -> topo2: add one line (two changes from topo0 in total)

--------------------------------------------------
File Structure
--------------------------------------------------

Each row corresponds to one successful scenario.

Column structure:

1. Pd_bus1, Qd_bus1  
2. Pd_bus2, Qd_bus2  
   ...  
3. Pd_busN, Qd_busN   ← (N = number of buses)

4. Pg_bus1  Qg_bus1
5. Pg_bus2  Qg_bus2
   ...  
6. Pg_busN  Qg_busN

7. Objective

Where:
- **Pd_busX**: Active power demand at bus X (0 if no load at that bus)
- **Qd_busX**: Reactive power demand at bus X (0 if no load at that bus)
- **Pg_busX**: Active power generation at bus X (0 if no generator at that bus)
- **Qg_busX**: Reactive power generation at bus X (0 if no generator at that bus)
- **Objective**: Total generation cost for the scenario

