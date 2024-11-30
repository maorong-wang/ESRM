learner             =           "GML"
DATASETS             =           ["cifar10", "cifar100", "tiny"]
# n_tasks             =           10
# proj_dim            =           256
optim               =           "SGD"
lr                  =           0.1
n_runs              =           3
batch_size          =           10
mem_batch_size      =           64
C_vals              =           [0]
B_vals              =           [0]
# L_init              =           [256]
# sigmas              =           [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.15]
var                 =           0.07
q_vals              =           [2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
nf                  =           64

for q in q_vals:
    for dataset in DATASETS:
        for B in B_vals:
            for C in C_vals:
                if dataset == 'cifar10':
                    n_tasks = 5
                    proj_dim = 128
                    L = 128
                    mem_size_values = [200, 500, 1000]
                elif dataset == 'cifar100':
                    n_tasks = 10
                    proj_dim = 256
                    L = 256
                    mem_size_values = [1000, 2000, 5000]
                elif dataset == 'tiny':
                    n_tasks = 100
                    proj_dim = 256
                    L = 256
                    # mem_size_values = [2000, 4000, 10000]
                    mem_size_values = [2000]
                for mem_size in mem_size_values:
                    tag = f"gml,q{q},{dataset}"
                    cfg_file = f"""
learner         :       {learner}
dataset         :       {dataset}
n_tasks         :       {n_tasks}
optim           :       {optim}
lr              :       {lr}
n_runs          :       {n_runs}
mem_size        :       {mem_size}
mem_batch_size  :       {mem_batch_size}
batch_size      :       {batch_size}
proj_dim        :       {proj_dim}
nf              :       {nf}
supervised      :       True
eval_mem        :       True
fixed_means     :       True
all_means       :       True
tot_classes_gml :       {L}
var             :       {var}
C               :       {C}
mem_iters       :       {q}
head_reg_coef   :       {B}
tag             :       {tag}
n_augs          :       1
n_mixup         :       0
"""                     
                    filename = f"{tag}_m{mem_size}mbs{mem_batch_size}sbs{batch_size}.yaml"
                    with open(filename, "w") as f:
                        f.write(cfg_file)
                    f.close
                    
                