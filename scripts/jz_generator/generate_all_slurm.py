import subprocess
import os
import glob

##############
# Run Params #
##############
dataset         =   "cifar100"
mem_size        =   2000
config_folder   =   "gml,v1.1"
root_dir        =   f"../../config/{config_folder}/"
data_dir        =   "../../data/"
#mi_values       =   [5, 10, 20, 30]
#results_root    =   f"./results/{config_folder}/{dataset}/"
results_root    =   f"./results/{config_folder}/"
#test_freq       =   50

###################
# Jean Zay params #
###################

time            =       '19:00:00'
module          =       'pytorch-gpu/py3/1.11.0'
gpu_type        =       'v100-32g'
qos             =       'qos_gpu-t3'
nodes           =       1
ntasks          =       1
gres            =       'gpu:1'
cpu_task        =       4
hint            =       'nomultithread'
logdir          =       './logs/'

# Slurm generation
configs_files = glob.glob(f"{root_dir}*yaml")
#configs_files = [f"{root_dir}2aug,{dataset},mi{mi}_m{mem_size}mbs200sbs100.yaml" for mi in mi_values]

for cfg in configs_files:
    sbatch_cmd =\
        f"""#!/bin/bash

#SBATCH --job-name={os.path.basename(cfg)[:-5]}
#SBATCH -C {gpu_type}                           # reserver des GPU 16 Go seulement# (specifier "-C v100-32g" pour des GPU à 32 Go)
#SBATCH --qos={qos}                             # QoS
#SBATCH --output={logdir}%j_{os.path.basename(cfg)[:-5]}.out  # fichier de sortie (%j = job ID)
#SBATCH --error={logdir}%j_{os.path.basename(cfg)[:-5]}.err   # fichier derreur (%j = job ID)
#SBATCH --time={time}                           # temps maximal dallocation "(HH:MM:SS)"
#SBATCH --nodes={nodes}                         # reserver 1 nœud
#SBATCH --ntasks={ntasks}                       # reserver 1 taches (ou processus MPI)
#SBATCH --gres={gres}                           # reserver 1 GPU
#SBATCH --cpus-per-task={cpu_task}              # reserver 10 CPU par tache (et memoire associee)
#SBATCH --hint={hint}                           # desactiver lhyperthreading
module purge                                    # nettoyer les modules herites par defaut
conda deactivate                                # desactiver les environnements herites par defaut
module load {module}                            # charger les modules
set -x                                          # activer lecho des commandes
    """
    pycmd =\
        f"python main.py --results-root {results_root} --data-root-dir {data_dir} --config ./config/{config_folder}/{os.path.basename(cfg)}"
    cmd = sbatch_cmd + "\n" + pycmd
    filename = f"./{os.path.basename(cfg)[:-5]}.sh"
    with open(filename, "w") as f:
        f.write(cmd)
    f.close()

    # Change permissions
    subprocess.call(['chmod', '0755', filename])
