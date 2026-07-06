from torch.utils.data import DataLoader
import torch 
import numpy as np
import pickle as pkl
from data_utils import *    
from ewc_utils import * 
import warnings
from utils import * 

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_heads(model: torch.nn.Module):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.heads

    return model.heads

def train_on_noise_model(
        noise_ckpt,
        seed=0,
        add_noise=True,
        n_epochs=None,
        eval_on_tst=True,
        init_eval=True,
        key="latest_noise",
        theta_lr=None,
        shuffle_noisy_data=False,
        rnd_noise=False,
        override_finetune=False,
        bs=128,
        tst_on_train=False,
        cil=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if "pkl" in noise_ckpt:
        with open(noise_ckpt, "rb") as file:
            noise_save_dict = pkl.load(file)
    else:
        noise_save_dict = noise_ckpt

    cont_method_args = (
        noise_save_dict["pretrained_ckpt"]["cont_method_args"]
    )

    model = create_load_add_head(
        **noise_save_dict["pretrained_ckpt"],
        load=True,
    )

    artifact_version = int(
        noise_save_dict.get("artifact_version", 0)
    )

    if artifact_version < 2:
        raise RuntimeError(
            "Legacy BrainWash artifact detected. "
            "Regenerate the attack with artifact_version >= 2."
        )

    if key == "latest_noise":
        head_state_key = "latest_noise_head_state"
        head_index_key = "latest_noise_head_index"
        head_epoch_key = "latest_noise_epoch"

    elif key == "noise":
        head_state_key = "best_noise_head_state"
        head_index_key = "best_noise_head_index"
        head_epoch_key = "best_noise_epoch"

    else:
        raise ValueError(
            f"Unsupported noise key: {key}. "
            "Expected 'latest_noise' or 'noise'."
        )

    required_keys = {
        key,
        head_state_key,
        head_index_key,
        head_epoch_key,
    }

    missing_keys = required_keys - set(noise_save_dict)

    if missing_keys:
        raise KeyError(
            "The attack artifact is missing required noise/head data: "
            f"{sorted(missing_keys)}"
        )

    task_heads = get_task_heads(model)

    head_index = int(
        noise_save_dict[head_index_key]
    )

    if head_index < 0 or head_index >= len(task_heads):
        raise RuntimeError(
            f"Saved head index {head_index} is invalid for a model "
            f"with {len(task_heads)} task heads."
        )

    expected_head_index = len(task_heads) - 1

    if head_index != expected_head_index:
        raise RuntimeError(
            f"Expected the incoming-task head at index "
            f"{expected_head_index}, but the artifact stores "
            f"index {head_index}."
        )

    selected_head_state = noise_save_dict[
        head_state_key
    ]

    task_heads[head_index].load_state_dict(
        selected_head_state,
        strict=True,
    )

    restored_head_state = (
        task_heads[head_index].state_dict()
    )

    for parameter_name, expected_tensor in selected_head_state.items():
        actual_tensor = (
            restored_head_state[parameter_name]
            .detach()
            .cpu()
        )

        expected_tensor = (
            expected_tensor
            .detach()
            .cpu()
        )

        if not torch.equal(
                actual_tensor,
                expected_tensor,
        ):
            raise RuntimeError(
                "Task-head restoration failed for parameter: "
                f"{parameter_name}"
            )

    print(
        "Exact attack-time task head restored:",
        f"noise_key={key},",
        f"head_index={head_index},",
        f"noise_epoch={noise_save_dict[head_epoch_key]}",
    )

    ds_dict = get_dataset_specs(
        **noise_save_dict["pretrained_ckpt"]
    )[0]
    ds_tst = ds_dict['test'][-1]
                                               
    
    ds_train = ds_dict['train'][-1]
    ds_train.data = ds_train.data[noise_save_dict['rnd_idx_train']]
    ds_train.targets = ds_train.targets[noise_save_dict['rnd_idx_train']]

    delta = noise_save_dict['delta']    
    
    if shuffle_noisy_data:
        suffle_idx = np.random.permutation(len(ds_train))   
        ds_train.data = ds_train.data[suffle_idx]
        ds_train.targets = ds_train.targets[suffle_idx]
        if rnd_noise == False:
            noise_data = noise_save_dict[key][suffle_idx]    
        else:
            noise_data = torch.rand_like(ds_train.data) * delta * 2 - delta
            
    else:
        if rnd_noise == False:
            noise_data = noise_save_dict[key]
        else:
            noise_data = torch.rand_like(ds_train.data) * delta * 2 - delta


    optim = create_optimizer(model, 'sgd', theta_lr)   
    print('optim: sgd') 

    if n_epochs == None:
        n_epochs = noise_save_dict['pretrained_ckpt']['n_epochs']

    if init_eval:
        acc_ = []
        for t in range(noise_save_dict['pretrained_ckpt']['task_num']+1):
            ds_tst = ds_dict['test'][t]
            dl_tst_tmp = DataLoader(ds_tst, batch_size=64, shuffle=True)
            acc = eval_dl(model, dl_tst_tmp, verbose=False, task_id=t)
            acc_.append(acc)

        acc_ = np.array(acc_)    

        with np.printoptions(precision=2, suppress=True):
            print(f'initial acc: {acc_}')

        avg_acc = noise_save_dict['pretrained_ckpt']['avg_acc']
        bwt = noise_save_dict['pretrained_ckpt']['bwt']
        
        print(f'initial acc mean: {avg_acc}')   
        print(f'initial bwt: {bwt}')

        print()
    
    
    model.train()

    if cont_method_args['method'] == 'finetune' or override_finetune:
        model = train_on_noise_model_finetune(model, noise_data, noise_save_dict, optim, ds_dict=ds_dict, ds_train=ds_train, ds_tst=ds_tst,
                                  add_noise=add_noise, n_epochs=n_epochs, eval_on_tst=eval_on_tst, bs=bs)


    elif cont_method_args['method'] == 'ewc':
        model = train_on_noise_model_ewc(model, noise_data, noise_save_dict, optim, ds_dict=ds_dict, ds_train=ds_train, ds_tst=ds_tst,
                                  add_noise=add_noise, n_epochs=n_epochs, 
                                  eval_on_tst=eval_on_tst, tst_on_train=tst_on_train, bs=bs, **cont_method_args)
        
        
        
  
    return model



def train_on_noise_model_finetune(model, noise_data, noise_save_dict, optim, ds_dict, ds_train, ds_tst, 
                             add_noise=True, n_epochs=None, eval_on_tst=True, bs=128, cil=False):
  
                                                
    loss_fn = torch.nn.CrossEntropyLoss()

    if len(ds_train) % bs == 0:
        num_of_iter = len(ds_train) // bs
    else:
        num_of_iter = len(ds_train) // bs + 1

    for epoch in range(n_epochs):
        model.train()
        for i in range(num_of_iter):
            tail_idx = min((i+1) * bs, len(ds_train))   
            x = ds_train.data[i*bs:tail_idx]    
            y = ds_train.targets[i*bs:tail_idx] 
            x, y = x.to(device), y.to(device)
            noise = noise_data[i*bs:tail_idx] 
            

            if add_noise:    
                noise = noise.to(device)    
                x_tilde = torch.clamp(x + noise, 0, 1)
            else:
                x_tilde = x

            y_hat = model(x_tilde)[-1]
            loss = loss_fn(y_hat, y)
           

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)   
            optim.step()

        if eval_on_tst:
            model.eval()
            if cil == False:
                acc_ = []
                for t in range(noise_save_dict['pretrained_ckpt']['task_num']+1):
                    ds_tst = ds_dict['test'][t]
                    dl_tst_tmp = DataLoader(ds_tst, batch_size=64, shuffle=True)
                    acc = eval_dl(model, dl_tst_tmp, verbose=False, task_id=t)
                    acc_.append(acc)

                prev_acc_mat = noise_save_dict['pretrained_ckpt']['acc_mat']   
                bwt = (acc_[:-1] - np.diagonal(prev_acc_mat)).mean()    


                with np.printoptions(precision=2, suppress=True):
                    print(f'epoch {epoch} acc: {np.array(acc_)}')
                    print(f'average acc up until: {np.mean(acc_[:-1])}') 
                    print(f'bwt: {bwt}')    
                    
                    print()
            else:
                tmp_ds = combine_ds_class_inc(noise_save_dict['pretrained_ckpt']['task_num']+1, ds_dict['test'])    
                tmp_dl_tst = DataLoader(tmp_ds, batch_size=bs, shuffle=False)    
                all_acc = acc_curr = eval_dl(model, tmp_dl_tst, verbose=False, class_inc=True)
                prev_acc_lst = []
                for t_id in range(noise_save_dict['pretrained_ckpt']['task_num']): 
                    tmp_ds = ds_dict['test'][t_id]  
                    tmp_dl_tst = DataLoader(tmp_ds, batch_size=bs, shuffle=False)    
                    acc_curr = eval_dl(model, tmp_dl_tst, verbose=False, class_inc=True)
                    prev_acc_lst.append(acc_curr)   
                    
                
                print(f'epoch {epoch} acc task on combined datasets over {noise_save_dict["pretrained_ckpt"]["task_num"]+1} tasks: {all_acc}')    
                print(f'epoch {epoch} acc task on individual datasets:\n {prev_acc_lst}')

                
            model.train()
  
    return model
