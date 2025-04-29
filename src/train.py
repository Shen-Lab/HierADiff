"""
    Training script

    TODO: add in sampling to evaluation every k epochs
    This can be accomplished by importing sample from sample
"""

import os
import sys
from collections import defaultdict
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import printt, print_res, get_optimizer
from utils import log, compute_rmsd
from sample import sample
from data import BindingDataset
from evaluation.compute_rmsd import evaluate_all_rmsds

import torch.distributed as dist

def train(train_loader, val_loader, model,
          writer, fold_dir, args, loaders_for_reverse_diffusion=None):
    # [amp]
    # scaler = torch.cuda.amp.GradScaler()

    amp = False

    # optimizer
    start_epoch, optimizer = get_optimizer(model, args, load_best=True)

    # validation
    best_loss = float("inf")
    best_metrics = {'rmsds_lt2': -1,
                    'rmsds_lt5': -1,
                    'rmsds_mean': float("inf"),
                    'rmsds_median': float("inf")}

    best_path = None
    best_epoch = start_epoch
    # logging
    log_path = os.path.join(fold_dir, "log.yaml")

    num_batches = start_epoch * (len(train_loader) // args.batch_size)
    ep_iterator = range(start_epoch, start_epoch+args.epochs)
    if not args.no_tqdm:
        ep_iterator = tqdm(ep_iterator,
                           initial=start_epoch,
                           desc="train epoch", ncols=50)
        
    for epoch in ep_iterator:
        # DDP; to keep the random seed same for multiple processes
        train_loader.sampler.set_epoch(epoch)

        writer.add_scalar("epoch", epoch, num_batches)
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
            print("batch!!!!!!!!!!!", batch['name'], "rank", args.rank)

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
                if writer is not None:
                    writer.add_scalar("gradient", grad_mean, num_batches)
                nn.utils.clip_grad_norm_(model.parameters(), 1) # TODO (AK): consider enabling this 
                optimizer.step()
                # # [amp]
                # scaler.step(optimizer)
                # scaler.update()

            # write to tensorboard
            num_batches += 1
            loss = loss.cpu().item()  # CPU for logging
            if num_batches % args.log_frequency == 0:
                for loss_type, loss in losses.items():
                    if loss is None:
                        continue
                    log_key = f"train_{loss_type}"
                    if writer is not None:
                        writer.add_scalar(log_key, loss.mean(), num_batches)

            if epoch % args.save_model_every_batch == 0 and (args.rank == 0):
                last_path = os.path.join(fold_dir, "model_last.pth")
                save_model(model, args, optimizer, last_path)

            # print("final in every training iteration", torch.cuda.memory_summary())


            # end of batch ========
            # force checkpoint saving + evaluation
            # VERY BAD BUG. just exits the training epoch after one single batch
            #if num_batches > 5000:
            #    break

        printt(f'\nEncountered {num_oom_errors} Out-of-memory errors during last epoch.')
        
        # evaluate (end of epoch)
        avg_train_loss, avg_train_score = train_epoch_end(num_batches,
                train_loader, model,
                log_path, fold_dir, args,
                max_batch=20, split="train")  # adjust based on batch size
        val_loss, avg_val_score = train_epoch_end(num_batches,
                val_loader, model, log_path, fold_dir, args, writer)
        printt('\nepoch', epoch)
        printt('train loss', avg_train_loss, args.metric, avg_train_score)
        printt('val loss', val_loss, args.metric, avg_val_score)

        # save latest model every e.g. 10 epochs
        # DDP
        # if epoch % args.save_model_every == 0 and epoch != 0:
        if epoch % args.save_model_every == 0 and epoch != 0 and (args.rank == 0):
            last_path = os.path.join(fold_dir, "model_last.pth")
            save_model(model, args, optimizer, last_path)
        # >>>
        #if avg_val_score < best_loss:
        #    best_loss = avg_val_score

        # save model
        #path_suffix = f"{num_batches}_{epoch}_{avg_val_score:.3f}_{val_loss:.3f}.pth"
        #if val_loss < best_loss:
        #    best_loss = val_loss
        #    best_epoch = epoch
        #    # save model ONLY IF best
        #    best_path = os.path.join(fold_dir, f"model_best_{path_suffix}")
        #    save_model(model, args, optimizer, best_path)

        ## check if out of patience
        #if epoch - best_epoch >= args.patience:
        #    break

        # DDP
        # if args.val_inference_freq is not None and epoch % args.val_inference_freq == 0: # and epoch != 0:

        # [validation flag]
        enable_val_flag = False
        if enable_val_flag and args.val_inference_freq is not None and epoch % args.val_inference_freq == 0 and args.rank == 0:
            # run reverse diffusion process
            # only use subset to run reverse diffusion process
            if args.num_inference_complexes is not None:
                data_list = loaders_for_reverse_diffusion["val"][:args.num_inference_complexes]
            else:
                data_list = loaders_for_reverse_diffusion["val"]

            samples_val = sample(data_list, model, args, epoch=epoch,
                                 visualize_first_n_samples=args.visualize_n_val_graphs,
                                 visualization_dir=args.visualization_path)

            meter = evaluate_all_rmsds(data_list, samples_val)
            ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized = meter.summarize()

            if writer is not None:
                for rd_key, rd_value in ligand_rmsd_summarized.items():
                    writer.add_scalar(f"val_lig_rmsd_{rd_key}", rd_value, num_batches)
                for rd_key, rd_value in complex_rmsd_summarized.items():
                    writer.add_scalar(f"val_complex_rmsd_{rd_key}", rd_value, num_batches)


            # save model if it improves rmsd
            if ligand_rmsd_summarized['mean'] < best_metrics['rmsds_mean'] and ligand_rmsd_summarized['median'] < best_metrics['rmsds_median']:
                best_metrics['rmsds_mean'] = ligand_rmsd_summarized['mean']
                best_metrics['rmsds_median'] = ligand_rmsd_summarized['median']
                best_epoch = epoch
                best_loss = val_loss

                path_suffix = f"{num_batches}_{epoch}_{best_metrics['rmsds_mean']:.3f}_{best_metrics['rmsds_median']:.3f}.pth"
                # save model ONLY IF best
                best_path = os.path.join(fold_dir, f"model_best_{path_suffix}")
                save_model(model, args, optimizer, best_path)
                
        # DDP
        if enable_val_flag and args.sample_train and args.val_inference_freq is not None and epoch % args.val_inference_freq == 0 and args.rank == 0:# and epoch != 0:
            # run reverse diffusion process
            # only use subset to run reverse diffusion process
            #if args.num_inference_complexes_train_data is not None:
            #    random_indices = np.random.choice(len(loaders_for_reverse_diffusion["train"]), size=args.num_inference_complexes_train_data, replace=False)
            #    data_list = loaders_for_reverse_diffusion["train"][random_indices]
            #else:
            #    data_list = loaders_for_reverse_diffusion["train"]

            if args.num_inference_complexes_train_data is not None:
                random_indices = np.random.choice(len(loaders_for_reverse_diffusion["train"]), size=args.num_inference_complexes_train_data, replace=False)

                data_list = BindingDataset(args, {}, apply_transform=False)
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

def save_model(model, args, optimizer, path):
    # DDP
    if args.num_gpu > 1 or args.local_rank >= 0:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({"model": state_dict,
                        "optimizer": optimizer.state_dict()}, path)
    printt(f"\nsaved model to {path}")


def train_epoch_end(num_batches, val_loader,
                    model, log_path, fold_dir, args,
                    writer=None,
                    max_batch=None, split="val"):
    """
        Evaluate at end of training epoch and write to log file
    """
    log_item = { "batch": num_batches }
    val_loss, avg_val_score = float("inf"), 0
    if len(val_loader) == 0:
        return val_loss, avg_val_score

    # run inference
    val_scores = evaluate(val_loader, model, writer, args,
                          max_batch=max_batch)
    metric = args.metric
    if metric not in val_scores:
        metric = list(val_scores)[0]
    avg_val_score = np.nanmean(val_scores[metric])
    for key, value in val_scores.items():
        try:
            log_item[f"{split}_{key}"] = value.mean()
        except:
            log_item[f"{split}_{key}"] = value
    print_res(log_item)

    # tensorboard
    if writer is not None:
        for key, value in log_item.items():
            if key not in ["batch"]:
                writer.add_scalar(key, value, num_batches)
    # append to YAML log
    log(log_item, log_path)
    # early stopping
    val_loss = val_scores["loss"]
    return val_loss, avg_val_score


def evaluate(val_loader, model, writer, args,
             max_batch=None):
    """
        @param (int) max_batch       number of batches to sample
    """
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
                            desc="evaluation",
                            leave=False, ncols=50)
        for batch_num, batch in iterator:
            # move to CUDA
            # DDP
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
                # DDP
                if args.num_gpu > 1 or args.local_rank >= 0:
                    losses = model.module.compute_loss(batch, output)
                else:
                    losses = model.compute_loss(batch, output)

                for loss_type, loss in losses.items():
                    all_losses[loss_type].append(loss.mean().item())

                for key in []:
                    if key in output:
                        all_output[key].append(output[key].detach().cpu())

                if max_batch is not None and batch_num >= max_batch:
                    break

            # catch OOM
            except RuntimeError as e:
                printt("RuntimeError in evaluate. ")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                # [Mod]
                # gc.collect()
                continue

    ######## evaluate metrics
    scores = {}
    # average loss over batches
    for loss_type, losses in all_losses.items():
        scores[loss_type] = torch.mean(torch.tensor(losses)).item()
    print_res(all_losses)

    # save outputs
    for key, value in all_output.items():
        try:
            value = torch.tensor(value).float()
        except ValueError:
            value = torch.cat(value)
        scores[key] = torch.mean(value).item()

    return scores


def evaluate_pose(data_list, samples_list):
    """
        Evaluate sampled pose vs. ground truth
    """
    all_rmsds = []
    rmsds_with_name = {}
    assert len(data_list) == len(samples_list)
    for true_graph, pred_graph in zip(data_list, samples_list):
        true_xyz = true_graph["ligand"].pos
        pred_xyz = pred_graph["ligand"].pos
        if true_xyz.shape != pred_xyz.shape:
            print(true_graph["name"], pred_graph["name"])
        assert true_xyz.shape == pred_xyz.shape
        rmsd = compute_rmsd(true_xyz, pred_xyz)
        all_rmsds.append(rmsd)
        rmsds_with_name[true_graph["name"]] = rmsd

    scores = {
        "rmsd": all_rmsds,
        "rmsds_with_name": rmsds_with_name,
    }

    return scores
