learner             =           "GML"
dataset             =           "cifar100"
n_tasks             =           10
proj_dim            =           256
optim               =           "SGD"
lr                  =           0.1
n_runs              =           5
mem_size            =           2000
batch_size          =           10
mem_batch_size      =           64
# C_vals              =           [0, 0.01, 0.05, 0.1]
C_vals              =           [0]
# B_vals              =           [0, 1e-4, 1e-3]
B_vals              =           [0]
L_init              =           [256]
sigmas              =           [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.15]
q_vals              =           [1]
cont_strat          =           'mem'

for q in q_vals:
    for B in B_vals:
        for C in C_vals:
            for L in L_init:
                for var in sigmas:
                    tag = f"gml,cont,c{C},b{B},L{L},v{var},q{q},{cont_strat},{dataset}"
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
supervised      :       True
eval_mem        :       True
fixed_means     :       True
all_means       :       True
tot_classes_gml :       {L}
cont_strat      :       {cont_strat}
var             :       {var}
C               :       {C}
mem_iters       :       {q}
head_reg_coef   :       {B}
tag             :       {tag}
"""                     
                    filename = f"{tag}_m{mem_size}mbs{mem_batch_size}sbs{batch_size}.yaml"
                    with open(filename, "w") as f:
                        f.write(cfg_file)
                    f.close
                    
                