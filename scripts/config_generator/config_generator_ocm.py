learner             =           "OCM"
DATASETS            =           ["cifar10", "cifar100", "tiny"]
batch_size          =           10
mem_batch_size      =           64
nf                  =           64
# blurry_scale        =           1500

for dataset in DATASETS:
    if dataset == 'cifar10':
        n_tasks = 5
        proj_dim = 128
        mem_size_values = [200, 500, 1000]
    elif dataset == 'cifar100':
        n_tasks = 10
        proj_dim = 128
        mem_size_values = [1000, 2000, 5000]
    elif dataset == 'tiny':
        n_tasks = 100
        proj_dim = 128
        mem_size_values = [2000, 5000, 10000]
    for mem_size in mem_size_values:
        tag = ""
        cfg_file = f"""
learner         :       {learner}
dataset         :       {dataset}
n_tasks         :       {n_tasks}
mem_size        :       {mem_size}
mem_batch_size  :       {mem_batch_size}
batch_size      :       {batch_size}
proj_dim        :       {proj_dim}
nf              :       {nf}
supervised      :       True
"""                     
        filename = f"{learner},{dataset},m{mem_size}mbs{mem_batch_size}sbs{batch_size}.yaml"
        with open(filename, "w") as f:
            f.write(cfg_file)
        f.close
                    
                