learner             =           "AUG"
# dataset             =           ["cifar10", "cifar100", "tiny"]
dataset             =           ["tiny"]
# n_tasks             =           5
# mi                  =           [1, 20, 100]
mi                  =           [20]
n_augs              =           [2,3]
n_mixup             =           [0]
n_cutmix            =           [0]
# n_styles            =           [1,2,3,4,5]
n_styles            =           [0]
style_samples       =           [1]
# n_cutmix             =           [1,2,3,4,5]
# style_samples       =           [1, 5, 10, 50]
# min_mix             =           [0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8,0.9,1]
min_mix             =           [0.5]
max_mix             =           [0.5, 0.6 ,0.7, 0.8, 0.9, 1]
# max_mix             =           [1]
optim               =           "SGD"
lr                  =           0.07
n_runs              =           5
mem_size            =           [2000]
batch_size          =           [100]
mem_batch_size      =           [200]

for ssample in style_samples:
    for style in n_styles:
        for mxmin in min_mix:
            for mxmax in max_mix:
                for naug in n_augs:
                    for nmix in n_mixup:
                        for ncmix in n_cutmix:
                            for i in mi:
                                for ms in mem_size:
                                    for bs in batch_size:
                                        for mbs in mem_batch_size:
                                            for ds in dataset:
                                                if ds == "tiny":
                                                    n_tasks = 20
                                                    ms = 2000
                                                elif ds == "cifar10":
                                                    n_tasks = 5
                                                    ms = 200
                                                elif ds == "cifar100":
                                                    n_tasks = 10
                                                    ms = 2000
                                                tag = f"{naug}aug,{ds},mi{i}"
                                                # tag = f"{naug}aug,cutmix{mxmin}_{mxmax},{ds},mi{i}"
                                                cfg_file = f"""
learner         :       {learner}
dataset         :       {ds}
mem_iters       :       {i}
n_augs          :       {naug}
n_tasks         :       {n_tasks}
optim           :       {optim}
lr              :       {lr}
n_runs          :       {n_runs}
mem_size        :       {ms}
mem_batch_size  :       {mbs}
batch_size      :       {bs}
tag             :       {tag}
n_mixup         :       {nmix}
n_cutmix        :       {ncmix}
min_mix         :       {mxmin}
max_mix         :       {mxmax}
n_styles        :       {style}
style_samples   :       {ssample}
"""                     
                                                filename = f"{tag}_m{ms}mbs{mbs}sbs{bs}.yaml"
                                                with open(filename, "w") as f:
                                                    f.write(cfg_file)
                                                f.close
                        
                