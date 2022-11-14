# Multi-view ICA for Shared and Individual Sources

Please run the following 

`pip install -r requirements.txt`

To run the experiment for comparing the MLE and our approach please run

`python run_bound_vs_mle.py`

The file `run_simulated.py` is the main file for all synthetic experiments. For example, 

`python run_simulated.py --m 10 --a 20 --c 5 --n 1000`

will run a benchmark experiment for a dataset generated according to a model with 10 views 20 sources per view 5 of which are shared. The results that are saved under the file `"log/out_{args.m}_{args.a}_{args.c}_lam_{args.lam}_noise_{args.noise}.pickle"` contain Amari distance and compute time for each method.  To get the results generated Fig 3a run

`./run_fig3a.ssh`

For Figure 3b

`./run_fig3b.ssh`

### Omics Data Fusion experiment
To fit our model to the omics data please run 

`python run_omics.py`

The inferred sources are stored in ``log/omics/S{i}_{sources}.csv`` where `i` refers to the dataset and `sources` the number of total sources specified in the file. The user can change the settings by changing the follwoing parameters (lines 14-17 in the file)

```   
a1 = 80 # total number of sources in dataset1
a2 = 80 # total number of sources in dataset2
c = 40 # shared sources
```
To run IVA-L-SOS with 100 sources per dataset on the omics data please run

`python run_iva_omics.py --k 100`

The results will be stored under `log/omics/res100.pickle`

To reproduce the glasso experiments run
```angular2html
python run_glasso.py --index 100

# for iva-l-sos 

python run_glasso.py --isIVA --index 100
```

Note that for this experiment you need to have R installed on your system as well the package `huge` (in the `R` terminal run `install.packages("huge")`).

The code for the other methods is taken from  https://github.com/hugorichard/multiviewica and  https://github.com/hugorichard/shica
