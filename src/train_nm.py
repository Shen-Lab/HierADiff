"""
    Training script

    TODO: add in sampling to evaluation every k epochs
    This can be accomplished by importing sample from sample
"""
import pickle

import os
import sys
from collections import defaultdict
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

#
from utils import printt, print_res, get_optimizer
#
from utils import log, compute_rmsd
#
from sample_nm import sample

#
from data import BindingDataset_nm

#
from evaluation.compute_rmsd_nm import evaluate_all_rmsds

#
from evaluation.compute_rmsd_nm import RMSDComputer

import torch.distributed as dist


# [nm]
def train(train_loader, val_loader, model,
          writer, fold_dir, args, loaders_for_reverse_diffusion=None):
    """ This is a train function with grad clip/amp and early stopping
    """
    # [amp]
    # scaler = torch.cuda.amp.GradScaler()

    amp = False

    # optimizer and start_epoch
    start_epoch, optimizer = get_optimizer(model, args, load_best=False)

    # validation
    # determine if we should load the best stats or init stats
    best_stats_path = os.path.join(fold_dir, "best_stats.pkl")
    best_metrics, best_epoch, best_loss, best_path  = load_stats(best_stats_path, start_epoch)

    printt(f"Training starts with current epoch: {start_epoch}, best epoch: {best_epoch}, best loss: {best_loss}")


    # logging
    log_path = os.path.join(fold_dir, "log.yaml")

    num_batches = start_epoch * (len(train_loader) // args.batch_size) # num_batches depend on the batch_size
    ep_iterator = range(start_epoch, start_epoch+args.epochs)
    if not args.no_tqdm:
        ep_iterator = tqdm(ep_iterator,
                           initial=start_epoch,
                           desc="train epoch", ncols=50)
        
    for epoch in ep_iterator:
        # DDP; to keep the random seed same for multiple processes
        train_loader.sampler.set_epoch(epoch)

        if args.rank == 0:
            writer.add_scalar("epoch", epoch, num_batches)
        dist.barrier()

        # start epoch!
        iterator = enumerate(train_loader)
        if not args.no_tqdm:
            iterator = tqdm(iterator,
                            total=len(train_loader),
                            desc="train batch",
                            leave=False, ncols=50)
        num_oom_errors = 0

        # [DEBUG]
        flag = True

        for batch_num, batch in iterator:

            # [DEBUG]
            # print("batch!!!!!!!!!!!", batch['name'], "rank", args.rank)

            # if batch['name'] != 'kf/1kf6.pdb3_5.dill' and flag:
            #     continue
            # else:
            #     flag = False



            # eval before training and start numbering epochs at 1

            # [!!!!!!!!!!!!!]
            # if epoch == 0:
            #     break

            if amp:
                pass
                # # move to CUDA
                # # DDP
                # if torch.cuda.is_available() and args.num_gpu == 1:
                #     # batch = batch.cuda()

                #     # DDP; Do not need this?
                #     # batch = batch.cuda(args.local_rank)
                #     batch = batch.to(args.device)

                #     # print("batch.device", batch.device)

                # # reset optimizer gradients
                # model.train()
                # optimizer.zero_grad()
                # torch.cuda.empty_cache()


                # # [!!!!!!!!]
                # # [Mod]
                # oom_flag = False

                # # forward pass
                # try:
                #     # [amp]
                #     with torch.cuda.amp.autocast():
                #         output = model(batch)
                #         # compute loss (modifies output in place)
                #         # DDP
                #         if args.num_gpu > 1 or args.local_rank >= 0:
                #             losses = model.module.compute_loss(batch, output)
                #         else:
                #             losses = model.compute_loss(batch, output)

                #         # print("before loss.backward", torch.cuda.memory_summary())

                #         # backpropagate
                #         loss = losses["loss"]

                #         # [Debug]
                #         print("loss device", loss.device)


                #         for name, param in model.named_parameters():
                #             if torch.any(torch.isnan(param)):
                #                 print(name, "nan")

                #     # loss.backward()
                #     scaler.scale(loss).backward()

                #     # print("after loss.backward", torch.cuda.memory_summary())


                # # [!!!!!!!!]
                # # catch OOM this is amazing !!!

                # except RuntimeError as e:
                #     num_oom_errors += 1
                #     # printt("RuntimeError. ")
                #     printt(e)
                #     if "out of memory" in str(e):
                #         oom_flag = True

                #     # [Mod]
                #     # for p in model.parameters():
                #     #     if p.grad is not None:
                #     #         del p.grad  # free some memory

                # if oom_flag:
                #     optimizer.zero_grad()
                #     # gc.collection()    
                #     torch.cuda.empty_cache()
                #     # print("final after oom", torch.cuda.memory_summary())
                #     continue

                # # for p in model.parameters():
                # #     if p.grad is not None:
                # #         del p.grad  # free some memory
                # # torch.cuda.empty_cache()
                # # # [Mod]
                # # gc.collect()




                # # log gradients
                # grads = []
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         #print(name, param.grad.norm())
                #         grads.append(param.grad.norm().item())
                # grad_mean = torch.tensor(grads).mean()
                # if writer is not None:
                #     writer.add_scalar("gradient", grad_mean, num_batches)
                # nn.utils.clip_grad_norm_(model.parameters(), 1) # TODO (AK): consider enabling this 
                # # optimizer.step()
                # # [amp]
                # scaler.step(optimizer)
                # scaler.update()
            
            else:

                # move to CUDA
                # DDP
                if torch.cuda.is_available() and args.num_gpu == 1:
                    # batch = batch.cuda()

                    # DDP; Do not need this?
                    # batch = batch.cuda(args.local_rank)
                    batch = batch.to(args.device)

                # reset optimizer gradients
                model.train()
                optimizer.zero_grad()
                torch.cuda.empty_cache()


                # [!!!!!!!!]
                # [Mod]
                oom_flag = False

                # forward pass
                try:
                    output = model(batch)
                    # compute loss (modifies output in place)
                    # DDP
                    if args.num_gpu > 1 or args.local_rank >= 0:
                        losses = model.module.compute_loss(batch, output)
                        
                    else:
                        losses = model.compute_loss(batch, output)
                        

                    # print("before loss.backward", torch.cuda.memory_summary())

                    # backpropagate
                    # # fix for unused params in the forward
                    # loss = losses["loss"] + 0. * sum(p.sum() for p in model.parameters())


                    loss = losses["loss"]

                    # # Add L1 regularization
                    # l1_lambda = args.l1_lambda 
                    # l1_norm = sum(param.abs().sum() for param in params if param.requires_grad)
                    # loss = loss + l1_lambda * l1_norm



                    ########### compute subloss gradients ###########

                    # model.zero_grad()

                    # # loss_types
                    # if args.flex_only:
                    #     loss_types = ["nm_r_loss","nm_l_loss"]
                    # else:
                    #     loss_types = ["tr_loss", "rot_loss","nm_r_loss","nm_l_loss"]

                    # for loss_type in loss_types: 
                    #     # check if losses[loss_type] requires grad
                    #     if not losses[loss_type].requires_grad:
                    #         continue
                    #     losses[loss_type].backward(retain_graph=True)

                    #     # Backpropagate subloss and store gradients
                    #     grad_subloss = [param.grad.norm().item() for name, param in model.named_parameters() if param.grad is not None]
                    #     grad_subloss_mean = torch.tensor(grad_subloss).mean()
                        
                    #     # log gradients
                    #     if writer is not None and args.rank == 0:
                    #         writer.add_scalar(f"{loss_type}_gradient", grad_subloss_mean, num_batches)
                    #     dist.barrier()

                    #     # Zero gradients again
                    #     model.zero_grad()

                    ########### compute subloss gradients ###########

                    for name, param in model.named_parameters():
                        if torch.any(torch.isnan(param)):
                            print(name, "nan")

                    loss.backward()

                    # scaler.scale(loss).backward()
                    # print("after loss.backward", torch.cuda.memory_summary())


                # [!!!!!!!!]
                # catch OOM !!!

                except RuntimeError as e:
                    num_oom_errors += 1
                    # printt("RuntimeError. ")
                    printt(e)
                    if "out of memory" in str(e):
                        oom_flag = True
                        print("OOM occurred!!!", e)
                        print("batch!!!!!!!!!!!", batch['name'], "rank", args.rank)
                    else:
                        raise e

                    # [Mod]
                    # for p in model.parameters():
                    #     if p.grad is not None:
                    #         del p.grad  # free some memory
                
                # if args.rank >= 0:
                #     oom_tensor = torch.tensor(1 if oom_flag else 0).to(args.device)
                #     dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
                #     oom_flag = oom_tensor.item() == 1

                if oom_flag:
                    optimizer.zero_grad()
                    # gc.collect()    
                    torch.cuda.empty_cache()
                    print("skipping batch due to oom")
                    # print("final after oom", torch.cuda.memory_summary())
                    continue

                # for p in model.parameters():
                #     if p.grad is not None:
                #         del p.grad  # free some memory
                # torch.cuda.empty_cache()
                # # [Mod]
                # gc.collect()




                # log gradients
                grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        #print(name, param.grad.norm())
                        grads.append(param.grad.norm().item())
                grad_mean = torch.tensor(grads).mean()

                if writer is not None and args.rank == 0:
                    writer.add_scalar("gradient", grad_mean, num_batches)
                dist.barrier()

                nn.utils.clip_grad_norm_(model.parameters(), 1) # TODO (AK): consider enabling this 
                optimizer.step()
                # # [amp]
                # scaler.step(optimizer)
                # scaler.update()

            # write to tensorboard
            num_batches += 1
            loss = loss.cpu().item()  # CPU for logging
            if num_batches % args.log_frequency == 0 and args.rank == 0:
                for loss_type, loss in losses.items():
                    if loss is None:
                        continue
                    log_key = f"train_{loss_type}"
                    if writer is not None:
                        writer.add_scalar(log_key, loss.mean(), num_batches)
            dist.barrier()

            if batch_num % args.save_model_every_batch == 0 and (args.rank == 0):
                last_path = os.path.join(fold_dir, "model_last.pth")
                save_model(model, args, optimizer, last_path)
            

            dist.barrier()

            # print("final in every training iteration", torch.cuda.memory_summary())


            # end of batch ========
            # force checkpoint saving + evaluation
            # VERY BAD BUG. just exits the training epoch after one single batch
            #if num_batches > 5000:
            #    break

        printt(f'\nEncountered {num_oom_errors} Out-of-memory errors during last epoch.')
        
        # update the epoch number after every epoch of training; used in the save_model
        args.epoch = epoch + 1
        
        # evaluate (end of epoch)
        avg_train_loss, avg_train_score = train_epoch_end(num_batches,
                train_loader, model,
                log_path, fold_dir, args,
                max_batch=20, split="train")  # adjust based on batch size
        # [?]
        val_loss, avg_val_score = train_epoch_end(num_batches,
                val_loader, model, log_path, fold_dir, args, writer)
        
        if args.rank == 0:
            printt('\nepoch', epoch)
            printt('train loss', avg_train_loss, args.metric, avg_train_score)

            # val_loss = torch.mean(sync_tensor_across_gpus(val_loss, args)) # [sync]
            printt('val loss', val_loss, args.metric, avg_val_score)



        # save latest model every e.g. 1 epochs
        # DDP
        # if epoch % args.save_model_every == 0 and epoch != 0:
        if args.rank == 0: # save every epoch

            # every epoch
            every_path = os.path.join(fold_dir, f"model_every_{epoch}.pth")
            save_model(model, args, optimizer, every_path)

            # last epoch
            last_path = os.path.join(fold_dir, "model_last.pth")
            save_model(model, args, optimizer, last_path)


        
        dist.barrier()

        
        # >>>
        #if avg_val_score < best_loss:
        #    best_loss = avg_val_score

        # early stopping depending on the validation loss
        if not args.val_sample_flag:
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch

                # save model
                path_suffix = f"{num_batches}_{epoch}_{avg_val_score:.3f}_{val_loss:.3f}.pth"
                if args.rank == 0:
                    # save model ONLY IF best
                    best_path = os.path.join(fold_dir, f"model_best_{path_suffix}")
                    save_model(model, args, optimizer, best_path, best_epoch)

                    # save best stats
                    best_stats_path = os.path.join(fold_dir, "best_stats.pkl")
                    save_stats(best_metrics, best_epoch, best_loss, best_path, best_stats_path)

                    printt(f"Update best model at epoch {epoch} with loss {val_loss:.3f}")
            
            dist.barrier()

            # check if out of patience
            # if yes, calculate the final metrics and break
            if False and epoch - best_epoch >= args.patience and args.rank == 0:
                # run reverse diffusion process
                # only use subset to run reverse diffusion process [TODO: modify this]
                if args.num_inference_complexes is not None:
                    data_list = BindingDataset_nm(args, {}, apply_transform=False)
                    data_list.data = [loaders_for_reverse_diffusion["val"].data[i] for i in range(args.num_inference_complexes)]
                    data_list.length = len(data_list.data)
                else:
                    data_list = loaders_for_reverse_diffusion["val"]
                
                samples_val = sample(data_list, model, args, epoch=epoch,
                                     visualize_first_n_samples=args.visualize_n_val_graphs,
                                     visualization_dir=args.visualization_path)
                
                meter = evaluate_all_rmsds(data_list, samples_val)
                printt(f'Final metrics after {epoch} epochs:')
                printt(meter.summarize())
            # dist.barrier()

            

            if epoch - best_epoch >= args.patience:
                printt(f"ran out of patience in {args.rank}")
                break

        # DDP
        # if args.val_inference_freq is not None and epoch % args.val_inference_freq == 0: # and epoch != 0:

        # [validation sample flag]
        if args.val_sample_flag and args.val_inference_freq is not None and epoch % args.val_inference_freq == 0: # and args.rank == 0:

            DEBUG = True

            
            # run reverse diffusion process
            # only use subset to run reverse diffusion process [TODO: modify this]
            if args.num_inference_complexes is not None:
                data_list_ori = loaders_for_reverse_diffusion["val"][:args.num_inference_complexes]
            else:
                data_list_ori = loaders_for_reverse_diffusion["val"]


            #
            # split the data_list into multiple parts by contruct different datasets
            # run inference on each part with one gpu
            
            num_replicas = args.world_size
            total_size = len(data_list_ori)
            indices = list(range(total_size))
            indices = indices[args.rank:total_size:num_replicas]
                
            data_list = BindingDataset_nm({}, args, {}, apply_transform=False)
            data_list.data = [data_list_ori.data[i] for i in indices]
            data_list.pdb_ids = list(data_list.data)
            data_list.length = len(data_list.data)

            

            if DEBUG:
                # print('total_size', total_size, 'rank: ', args.rank)
                # print('indices', indices, 'rank: ', args.rank)
                pass


            if DEBUG:
                # print('data_list', data_list, 'rank: ', args.rank)
                pass


            
            samples_val = sample(data_list, model, args, epoch=epoch,
                                 visualize_first_n_samples=args.visualize_n_val_graphs,
                                 visualization_dir=args.visualization_path)
    

            meter = evaluate_all_rmsds(data_list, samples_val)

            #
            # gather the attributes of meter from multiple gpus (first convert them to tensors)

            complex_rmsd_tensor = torch.tensor(meter.complex_rmsd_list).to(args.device)
            ligand_rmsd_tensor = torch.tensor(meter.ligand_rmsd_list).to(args.device)
            interface_rmsd_tensor = torch.tensor(meter.interface_rmsd_list).to(args.device)
            recovery_rmsd_prec_tensor = torch.tensor(meter.recovery_rmsd_prec_list).to(args.device)

            if DEBUG:
                # print("rank: ", args.rank, "complex_rmsd_tensor", complex_rmsd_tensor)
                pass





            
            # first construct a meter object with the attributes from all gpus
            complex_rmsd_tensor = all_gather(complex_rmsd_tensor, args)
            ligand_rmsd_tensor = all_gather(ligand_rmsd_tensor, args)
            interface_rmsd_tensor = all_gather(interface_rmsd_tensor, args)
            recovery_rmsd_prec_tensor = all_gather(recovery_rmsd_prec_tensor, args)


            # then call summarize on the meter object (modify the next line to do this)

            if args.rank == 0:

                meter_all = RMSDComputer()

                meter_all.complex_rmsd_list = complex_rmsd_tensor.cpu() # now tensor and ok; only used by the summarize method
                meter_all.ligand_rmsd_list = ligand_rmsd_tensor.cpu() # now tensor and ok; only used by the summarize method
                meter_all.interface_rmsd_list = interface_rmsd_tensor.cpu() # now tensor and ok; only used by the summarize method
                meter_all.recovery_rmsd_prec_list = recovery_rmsd_prec_tensor.cpu() # now tensor and ok; only used by the summarize method


                if DEBUG:
                    # print("rank: ", args.rank, "meter_all.complex_rmsd_list", meter_all.complex_rmsd_list)
                    pass

            # dist.barrier() 

            
                ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized, recovery_rmsd_prec_summarized = meter_all.summarize()

                if writer is not None: # this is gauranteed as rank==0
                    for rd_key, rd_value in ligand_rmsd_summarized.items():
                        writer.add_scalar(f"val_lig_rmsd_{rd_key}", rd_value, num_batches)
                    for rd_key, rd_value in complex_rmsd_summarized.items():
                        writer.add_scalar(f"val_complex_rmsd_{rd_key}", rd_value, num_batches)
                    if recovery_rmsd_prec_summarized is not None:
                        for rd_key, rd_value in recovery_rmsd_prec_summarized.items():
                            writer.add_scalar(f"val_recovery_rmsd_prec_{rd_key}", rd_value, num_batches)


                # save model if it improves rmsd
                if ligand_rmsd_summarized['mean'] < best_metrics['rmsds_mean'] and ligand_rmsd_summarized['median'] < best_metrics['rmsds_median']:
                    best_metrics['rmsds_mean'] = ligand_rmsd_summarized['mean']
                    best_metrics['rmsds_median'] = ligand_rmsd_summarized['median']
                    best_epoch = epoch
                    best_loss = val_loss

                    path_suffix = f"{num_batches}_{epoch}_{best_metrics['rmsds_mean']:.3f}_{best_metrics['rmsds_median']:.3f}.pth"
                    # save model ONLY IF best
                    best_path = os.path.join(fold_dir, f"model_best_{path_suffix}")
                    save_model(model, args, optimizer, best_path, best_epoch) # [? rank==0?]

                    # save best stats
                    best_stats_path = os.path.join(fold_dir, "best_stats.pkl")
                    save_stats(best_metrics, best_epoch, best_loss, best_path, best_stats_path)
        
            dist.barrier()        
                
        # DDP
        # [!!!!!!!]
        if False and enable_val_flag and args.sample_train and args.val_inference_freq is not None and epoch % args.val_inference_freq == 0 and args.rank == 0:# and epoch != 0:
            # run reverse diffusion process
            # only use subset to run reverse diffusion process
            #if args.num_inference_complexes_train_data is not None:
            #    random_indices = np.random.choice(len(loaders_for_reverse_diffusion["train"]), size=args.num_inference_complexes_train_data, replace=False)
            #    data_list = loaders_for_reverse_diffusion["train"][random_indices]
            #else:
            #    data_list = loaders_for_reverse_diffusion["train"]

            if args.num_inference_complexes_train_data is not None:
                random_indices = np.random.choice(len(loaders_for_reverse_diffusion["train"]), size=args.num_inference_complexes_train_data, replace=False)

                data_list = BindingDataset_nm(args, {}, apply_transform=False)
                data_list.data = [loaders_for_reverse_diffusion["train"].data[i] for i in random_indices]
                data_list.length = len(data_list.data)
            else:
                data_list = loaders_for_reverse_diffusion["train"]

            samples_val = sample(data_list, model, args, epoch=epoch,
                                 visualize_first_n_samples=args.visualize_n_val_graphs,
                                 visualization_dir=args.visualization_path)

            print(f'on {len(samples_val)} training samples:')
            meter = evaluate_all_rmsds(data_list, samples_val)
            ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized = meter.summarize()

            if writer is not None:
                for rd_key, rd_value in ligand_rmsd_summarized.items():
                    writer.add_scalar(f"train_lig_rmsd_{rd_key}", rd_value, num_batches)
                for rd_key, rd_value in complex_rmsd_summarized.items():
                    writer.add_scalar(f"train_complex_rmsd_{rd_key}", rd_value, num_batches)


        # check if out of patience
        if epoch - best_epoch >= args.patience:
            break


        # end of epoch ========
    # end of all epochs ========

    return best_loss, best_epoch, best_path
    #return best_score, best_epoch, best_path

# [self]
def save_stats(best_metrics, best_epoch, best_loss, best_model_path, save_best_stats_path):
    '''
    Save best stats to file.
    '''
    
    with open(save_best_stats_path, 'wb') as f:
        pickle.dump({'best_metrics': best_metrics, 'best_epoch': best_epoch, 'best_loss': best_loss, 'best_model_path': best_model_path}, f)


def load_stats(best_stats_path, start_epoch):
    '''
    Load best stats from file if file exists
    Use initial value including start_epoch as the best_epoch if file does not exist

    '''

    if not os.path.exists(best_stats_path):
        # create the pickle file to save the current stats
        with open(best_stats_path, 'wb') as f:
            pickle.dump({'best_metrics': {'rmsds_lt2': -1,
                                         'rmsds_lt5': -1,
                                         'rmsds_mean': float("inf"),
                                         'rmsds_median': float("inf")},
                         'best_epoch': start_epoch,
                         'best_loss': float("inf"),
                         'best_model_path': None}, f)
        
        return {'rmsds_lt2': -1,
                'rmsds_lt5': -1,
                'rmsds_mean': float("inf"),
                'rmsds_median': float("inf")}, start_epoch, float("inf"), None
    
    with open(best_stats_path, 'rb') as f:
        di = pickle.load(f)
    
    return di['best_metrics'], di['best_epoch'], di['best_loss'], di['best_model_path']



# [nm]
def save_model(model, args, optimizer, path, epoch=None):

    if not epoch: # if no epoch is specified, use the current epoch (updated only at the end of each epoch)
        epoch = args.epoch
    
    # DDP
    if args.num_gpu > 1 or args.local_rank >= 0:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({"model": state_dict,
                "optimizer": optimizer.state_dict(),"epoch": epoch}, path)
    printt(f"\nsaved model to {path}")


# [nm]
def train_epoch_end_old(num_batches, val_loader,
                    model, log_path, fold_dir, args,
                    writer=None,
                    max_batch=None, split="val"):
    """
        Evaluate at end of training epoch and write to log file
    """

    DEBUG = False

    log_item = { "batch": num_batches }
    
    if len(val_loader) == 0:
        return float("inf"), 0

    # run inference
    val_scores = evaluate(val_loader, model, writer, args,
                          max_batch=max_batch,split=split)
    if DEBUG:
        print(val_scores)
    #
    # Sync losses and metrics across GPUs
    synced_val_scores = {}
    for key, value in val_scores.items():
        synced_val_scores[key] = torch.mean(sync_tensor_across_gpus(val_scores[key], args)).item()
    #
    # Pick the primary metric for validation
    metric = args.metric if args.metric in synced_val_scores else list(synced_val_scores)[0]
    avg_val_score = np.nanmean(synced_val_scores[metric])


    # # sync across gpus and average
    # new_val_scores = {}
    # for key, value in val_scores.items():
    #     new_val_scores[key] = torch.mean(sync_tensor_across_gpus(val_scores[key], args)).item() # [sync]
    # val_scores = new_val_scores


    # metric = args.metric
    # if metric not in val_scores:
    #     metric = list(val_scores)[0]
    # avg_val_score = np.nanmean(val_scores[metric])

    # for key, value in val_scores.items():
    #     try:
    #         log_item[f"{split}_{key}"] = value.mean()
    #     except:
    #         log_item[f"{split}_{key}"] = value

    #
    # Log the validation metrics for each key
    for key, value in synced_val_scores.items():
        log_item[f"{split}_{key}"] = value
    

    if args.rank == 0:
        # averaged values among all ranks
        # batch number only on rank 0
        print_res(log_item)


    # tensorboard
    if writer is not None and args.rank == 0:
        for key, value in log_item.items():
            if key != "batch":
                writer.add_scalar(key, value, num_batches)

    dist.barrier()
    

    # append to YAML log
    log(log_item, log_path)
    # early stopping
    val_loss = val_scores["loss"]

    val_loss = synced_val_scores.get("loss", float("inf"))


    return val_loss, avg_val_score


def train_epoch_end(num_batches, val_loader, model, log_path, fold_dir, args, writer=None, max_batch=None, split="val"):
    """
    Evaluate at end of training epoch, ensure global average loss across all GPUs, and write to log file.
    
    Args:
        num_batches: Current batch number.
        val_loader: DataLoader for the validation or test set.
        model: The model to evaluate.
        log_path: Path to the log file.
        fold_dir: Directory for saving the model checkpoints.
        args: Argument parser containing settings.
        writer: Tensorboard writer (optional).
        max_batch: Max number of batches to evaluate (for quick tests).
        split: Name of the dataset split, e.g., 'val' or 'test'.
    """

    DEBUG = False

    log_item = {"batch": num_batches}

    if len(val_loader) == 0:
        return float("inf"), 0

    # Run inference and get the total loss and total samples for each GPU
    val_loss_dict, total_samples = evaluate(val_loader, model, writer, args, max_batch=max_batch, split=split)

    # Sync each loss type and total_samples across all GPUs using sync_tensor_across_gpus
    global_loss_dict = {}
    for loss_type, loss_value in val_loss_dict.items():
        loss_tensor = sync_tensor_across_gpus(torch.tensor([loss_value], device=args.device), args)
        global_loss_dict[loss_type] = loss_tensor.sum().item()

    total_samples_tensor = sync_tensor_across_gpus(torch.tensor([total_samples], device=args.device), args)
    global_total_samples = total_samples_tensor.sum().item()

    # Calculate the global average for each loss type
    global_avg_loss_dict = {loss_type: global_loss / global_total_samples for loss_type, global_loss in global_loss_dict.items()}

    # Pick the primary metric for validation
    metric = args.metric if args.metric in global_avg_loss_dict else list(global_avg_loss_dict)[0]
    global_avg_loss_metric = global_avg_loss_dict[metric]


    # Log results for rank 0
    if args.rank == 0:
        for loss_type, global_avg_loss in global_avg_loss_dict.items():
            log_item[f"{split}_{loss_type}"] = global_avg_loss
            print_res(log_item)

            if writer is not None:
                writer.add_scalar(f"{split}_{loss_type}", global_avg_loss, num_batches)

        # Log all losses into YAML
        log(log_item, log_path)

    # Barrier to ensure all processes are synced
    dist.barrier()

    val_loss = global_avg_loss_dict.get("loss", float("inf"))

    return val_loss, global_avg_loss_metric  # Return the average loss and average loss for the metric


# [nm]
def evaluate_old(val_loader, model, writer, args,
             max_batch=None, split="val"):
    """
        @param (int) max_batch       number of batches to sample
    """

    DEBUG = False

    all_output = defaultdict(list)
    all_losses = defaultdict(list)

    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()

        # loop through all batches
        iterator = enumerate(val_loader)
        if not args.no_tqdm:
            iterator = tqdm(iterator,
                            total=len(val_loader),
                            desc=f"evaluation on {split}",
                            leave=False, ncols=50)
            
        for batch_num, batch in iterator:

            # move to CUDA
            if torch.cuda.is_available() and args.num_gpu == 1:
                # batch = batch.cuda()
                # DDP; Do not need this?
                # batch = batch.cuda(args.local_rank)
                batch = batch.to(args.device)

            # if we evaluate via poses, no need to compute fake loss
            if args.mode == "test" and not args.debug:
                printt("Skipping to pose")
                break
        
            # model predictions
            try:
                output = model(batch)

                # Compute loss
                # DDP
                if args.num_gpu > 1 or args.local_rank >= 0:
                    losses = model.module.compute_loss(batch, output)
                else:
                    losses = model.compute_loss(batch, output)

                if DEBUG:
                    print("losses", losses)

                # Accumulate losses
                for loss_type, loss in losses.items():
                    all_losses[loss_type].append(loss.mean().item())

                # Optionally process outputs (e.g., for visualization)
                for key in []: # Modify this if you want to process specific outputs
                    if key in output:
                        all_output[key].append(output[key].detach().cpu())

                if max_batch is not None and batch_num >= max_batch:
                    break

            # catch OOM
            except RuntimeError as e:
                printt("RuntimeError in evaluate. ")
                printt(e)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                # [Mod]
                # gc.collect()
                continue
    
    if DEBUG:
        print("all_losses", all_losses)


    ######## evaluate metrics
    scores = {}
    # average loss over batches
    for loss_type, losses in all_losses.items():
        scores[loss_type] = torch.mean(torch.tensor(losses)).item()
    # print_res(all_losses)

    # save outputs
    for key, value in all_output.items():
        try:
            value = torch.tensor(value).float()
        except ValueError:
            value = torch.cat(value)
        scores[key] = torch.mean(value).item()

    return scores


def evaluate(val_loader, model, writer, args, max_batch=None, split="val"):
    """
    Evaluate the model on the validation/test set, ensuring a global average of the loss
    across all samples, accounting for uneven batch sizes and multiple GPUs.
    
    Args:
        val_loader: DataLoader for validation or test set.
        model: The model to evaluate.
        writer: Tensorboard writer (optional).
        args: Argument parser containing evaluation settings.
        max_batch: Max number of batches to evaluate (for quick tests).
        split: Name of the dataset split, e.g., 'val' or 'test'.
    
    Returns:
        total_loss_dict: A dictionary of summed losses for each loss type across all samples in this GPU.
        total_samples: The total number of samples processed in this GPU.
    """

    total_loss_dict = defaultdict(float)  # Sum of all losses for each loss type
    total_samples = 0  # Total number of samples processed
    all_output = defaultdict(list)

    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        torch.cuda.empty_cache()  # Free up unused memory

        iterator = enumerate(val_loader)
        if not args.no_tqdm:
            iterator = tqdm(iterator, total=len(val_loader), desc=f"evaluating on {split}", leave=False, ncols=50)

        for batch_num, batch in iterator:
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch)
            total_samples += batch_size  # Track total samples processed

            # move to CUDA
            if torch.cuda.is_available():
                batch = batch.to(args.device)

            try:
                # Run model forward pass
                output = model(batch)

                # Compute loss
                if args.num_gpu > 1 or args.local_rank >= 0:
                    losses = model.module.compute_loss(batch, output)
                else:
                    losses = model.compute_loss(batch, output)

                # Accumulate weighted losses for each loss type
                for loss_type, loss in losses.items():
                    total_loss_dict[loss_type] += loss.mean().item() * batch_size  # Multiply by batch size for proper weighting

                # Optionally process outputs (e.g., for visualization)
                for key in []:  # Modify this if you want to process specific outputs
                    if key in output:
                        all_output[key].append(output[key].detach().cpu())

                # Break if max_batch limit is reached
                if max_batch is not None and batch_num >= max_batch:
                    break

            # catch OOM
            except RuntimeError as e:
                printt("RuntimeError in evaluate. ")
                printt(e)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue

    return total_loss_dict, total_samples



# [nm]
def evaluate_pose(data_list, samples_list):
    """
        Evaluate sampled pose vs. ground truth
    """
    all_rmsds = []
    rmsds_with_name = {}
    assert len(data_list) == len(samples_list)
    for true_graph, pred_graph in zip(data_list, samples_list):
        true_xyz = true_graph["ligand"].pos_b
        rec_xyz = true_graph["receptor"].pos_b

        pred_lig_xyz = pred_graph["ligand"].pos
        pred_rec_xyz = pred_graph["receptor"].pos

        meter = RMSDComputer()

        rmsd = meter.update_ligand_rmsd(pred_lig_xyz.numpy(), true_xyz.numpy(), pred_rec_xyz.numpy(), rec_xyz.numpy())

        all_rmsds.append(rmsd)
        rmsds_with_name[true_graph["name"]] = rmsd

    scores = {
        "rmsd": all_rmsds,
        "rmsds_with_name": rmsds_with_name,
    }

    return scores

# [self]
def all_gather(q, args):
    """
    Gathers tensor arrays of different lengths across multiple gpus (only 1-D tensors are supported for now)
    
    Parameters
    ----------
        q : tensor array
        args : args including args.world_size
        
    Returns
    -------
        torch.cat(all_q) : cat of list of gathered tensor arrays from all the gpus

    """
    ws = args.world_size


    local_size = torch.tensor(q.size(), device=q.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size.item() - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, device=q.device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])
    return torch.cat(all_qs)


# [self]
def sync_tensor_across_gpus(t, args):
    ''' t has to be with size (n,); t might have different values on different gpus.
        the result is the cat of all tensors from all gpus.
    '''
    if t is None:
        return None

    # check if t is a tensor
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    
    t = t.reshape(-1).to(args.device)

    group_size = args.world_size
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)  

    # print('rank:', args.rank, 'gather_t_tensor', gather_t_tensor)
    return torch.cat(gather_t_tensor, dim=0)

# [self]
def broadcast_tensor(tensor, src=0):
    """ Broadcasts a tensor from the source to all other processes. """

    # convert tensor to torch.tensor
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)

    dist.broadcast(tensor, src)