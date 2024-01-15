import wandb

class WandbUploader(object):
    def __init__(self,args,trainer):
        
        name = trainer + "{}_{}".format(
            args["model_name"],
            args["fed_algo"]
        )
        
        job_ms = ""
        
        if args["fixed_memory"]:
            name += "_fixed_" + str(args["memory_size_per_class"])
            job_ms = "_fixed_" + str(args["memory_size_per_class"])
        else:
            name += "_ms" + str(args["memory_size"])
            job_ms = "_ms" + str(args["memory_size"])
            
        if str(args["model_name"]) == "ours":
            name += "_to" + str(args["temperature_old"])
            name += "_tn" + str(args["temperature_new"])
            name += "_alpha" + str(args["alpha"])
            
        
        self.uploader = wandb.init(entity='xxx', project='xxx',
                    group = "_ds" + args["dataset"] + \
                        "_ct" + args["convnet_type"] + \
                        "_da" + str(args["dirichlet_alpha"]) + \
                        "_dp" + str(args["dataset_partition"]) + \
                        "_ic" + str(args["init_cls"]) + \
                        "_in" + str(args["increment"]) + \
                        "_lcpti" + str(args["local_classnum_per_task_init"]) + \
                        "_lcpt" + str(args["local_classnum_per_task"]) + \
                        "_tc" + str(args["total_clients"]) + \
                        "_sc" + str(args["select_clientnum"]),
                    name=name,
                    job_type= trainer + args["prefix"] + \
                        job_ms + \
                        "_seed" + str(args["seed"]) + \
                        "_bs" + str(args["batch_size"]) + \
                        "_lr" + str(args["lr"]) + \
                        "_arpt" + str(args["aggregation_round_per_task"]) + \
                        "_eps" + str(args["epoches"]),
                )
        self.uploader.config.update(args)