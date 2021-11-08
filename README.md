# PV-PPSO
Code for the paper: *Highly efficient photovoltaic parameter estimation using parallel particle swarm optimization on a GPU*. Accepted by the 30th International Symposium on Industrial Electronics (ISIE), held 20-23 June 2021 in Kyoto, Japan.

## View files
You can view the notebook files in the [notebook](./notebook) directory online.
## How to run
### Install Julia (if not yet)
This code is implemented with the [Julia programming language](https://julialang.org/). Please first install Julia. We worked with Julia v1.6.0-rc1, but other versions like v1.5.3 and higher versions should also work.
### Run the code with the steps below
1. Download or clone this repository to your local machine.
2. Launch [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) from the repository directory, which has been installed automatically during Julia installation.
3. In Julia REPL:
	- type `]` to enter package mode
	- type `activate .` to activate the current environment of PV-PPSO
	- type `instantiate` to install required packages of current project
	- press backspace until leaving the package mode 
4. Type the following in the Julia REPL to start Jupyter notebook for Julia ([documentation](https://github.com/JuliaLang/IJulia.jl))
	```julia
	using IJulia
	notebook(dir=pwd())
	```
5. In the Jupyter notebook shown in your browser, go into the `notebook` subfolder to run notebooks.


## How to cite
```bib
@INPROCEEDINGS{Gao-Highly,  
author={Gao, Shuhua and Xiang, Cheng and Lee, Tong Heng},  
booktitle={2021 IEEE 30th International Symposium on Industrial Electronics (ISIE)},   
title={Highly efficient photovoltaic parameter estimation using parallel particle swarm optimization on a GPU},   
year={2021},  
volume={},  
number={},  
pages={1-7},  
doi={10.1109/ISIE45552.2021.9576495}
}
```
