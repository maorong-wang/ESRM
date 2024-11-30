learner             =           "LUMP"
dataset             =           ["cifar10", "cifar100", "tiny"]
# dataset             =           ["tiny"]
mi                  =           [1]
optim               =           "SGD"
lr                  =           0.07
n_runs              =           5
mem_size            =           [2000]
batch_size          =           [100]
mem_batch_size      =           [200]

for i in mi:
    for ms in mem_size:
        for bs in batch_size:
            for mbs in mem_batch_size:
                for ds in dataset:
                    if ds == "tiny":
                        n_tasks = 20
                        ms = 5000
                    elif ds == "cifar10":
                        n_tasks = 5
                        ms = 500
                    elif ds == "cifar100":
                        n_tasks = 10
                        ms = 5000
                    tag = f"lump,{ds},mi{i}"
                    cfg_file = f"""
learner         :       {learner}
dataset         :       {ds}
mem_iters       :       {i}
n_tasks         :       {n_tasks}
optim           :       {optim}
lr              :       {lr}
n_runs          :       {n_runs}
mem_size        :       {ms}
batch_size      :       {bs}
mem_batch_size  :       {mbs}
tag             :       {tag}
"""                     
                    filename = f"{tag}_m{ms}mbs{mbs}sbs{bs}.yaml"
                    with open(filename, "w") as f:
                        f.write(cfg_file)
                    f.close
                        
                