learner             =           "ER"
DATASETS             =           ["cifar10", "cifar100", "tiny"]
batch_size          =           10
mem_batch_size      =           64
nf                  =           64
training_type_v     =           ['inc', 'blurry']
blurry_scale        =           1500

for training_type in training_type_v:
    for dataset in DATASETS:
        if dataset == 'cifar10':
            n_tasks = 5
            mem_size_values = [200, 500, 1000]
        elif dataset == 'cifar100':
            n_tasks = 10
            mem_size_values = [1000, 2000, 5000]
        elif dataset == 'tiny':
            n_tasks = 100
            mem_size_values = [2000, 5000, 10000]
        for mem_size in mem_size_values:
            tag = ""
            if training_type == 'blurry':
                cfg_file = f"""
learner         :       {learner}
optim           :       SGD
learning_rate   :       0.01
momentum        :       0.9
weight_decay    :       0.0001
dataset         :       {dataset}
n_tasks         :       {n_tasks}
mem_size        :       {mem_size}
mem_batch_size  :       {mem_batch_size}
batch_size      :       {batch_size}
nf              :       {nf}
training_type   :       {training_type}
blurry_scale    :       {blurry_scale}
"""                     
                filename = f"{learner},{dataset},m{mem_size}mbs{mem_batch_size}sbs{batch_size},blurry{blurry_scale}.yaml"
                with open(filename, "w") as f:
                    f.write(cfg_file)
                f.close
            else:
                cfg_file = f"""
learner         :       {learner}
optim           :       SGD
learning_rate   :       0.01
momentum        :       0.9
weight_decay    :       0.0001
dataset         :       {dataset}
n_tasks         :       {n_tasks}
mem_size        :       {mem_size}
mem_batch_size  :       {mem_batch_size}
batch_size      :       {batch_size}
nf              :       {nf}
training_type   :       {training_type}
"""                     
                filename = f"{learner},{dataset},m{mem_size}mbs{mem_batch_size}sbs{batch_size}.yaml"
                with open(filename, "w") as f:
                    f.write(cfg_file)
                    
                