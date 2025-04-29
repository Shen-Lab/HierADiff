"""
    Inference script
"""

import os
import gc
import sys
import copy

import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.data import Batch
from scipy.spatial.transform import Rotation as R


from utils import printt
from geom_utils import set_time_flexSDE_Rt, NoiseTransform_flexSDE_Rt
from geom_utils import axis_angle_to_matrix

from evaluation.compute_rmsd_nm import evaluate_all_rmsds, evaluate_all_ori_rmsds


EPS = 1e-6


DEBUG_inv = False

DEBUG_steps_80 = False # True # 

DEBUG_no_random = True # False #



# [nm]
def sample_flexSDE_Rt(data_list, model, args, epoch=0, visualize_first_n_samples=0,
           visualization_dir="./visualization", in_batch_size=None, output_coeffs=False):
    """
        Run reverse process
    """
    DEBUG = False

    args.num_steps = 40

    if in_batch_size is None:
        in_batch_size = args.batch_size
        
    # switch to eval mode
    model.eval()

    # stores various noise-related utils
    transform = NoiseTransform_flexSDE_Rt(args)

    # diffusion timesteps [N/N, N-1/N, ..., 1/N]
    timesteps = get_timesteps(args.num_steps)

    

    # # Prepare for visualizations
    # visualize_first_n_samples = min(visualize_first_n_samples, len(data_list))
    # graph_gts = [data_list[i] for i in range(visualize_first_n_samples)]
    # visualization_values = [data_list.get_visualization_values(index=i) for i in range(visualize_first_n_samples)]
    # four_letter_pdb_names = [get_four_letters_pdb_identifier(graph_gt.name) for graph_gt in graph_gts]
    # visualization_dirs = create_visualization_directories(visualization_dir, epoch, four_letter_pdb_names)

    # sde, rand_ub
    args.starting_from_random_flexed_unbound = True #False #  !!!!
    args.ode = False #True #  !!!!

    # # ode, ub
    # args.starting_from_random_flexed_unbound = False #True #  !!!!
    # args.ode = True #False #  !!!!

    # randomize original position and COPY data_list
    data_list, P_lig_dict, lambda_lig_dict, sampled_y_lig_dict, P_rec_dict, lambda_rec_dict, sampled_y_rec_dict = randomize_position_AND_use_unbound(data_list, args)

    if torch.cuda.is_available() and args.num_gpu == 1:
        # P_lig_dict, lambda_lig_dict, sampled_y_lig_dict, P_rec_dict, lambda_rec_dict, sampled_y_rec_dict
        P_lig_dict = {key: P_lig_dict[key].cuda(args.gpu) for key in P_lig_dict}
        lambda_lig_dict = {key: lambda_lig_dict[key].cuda(args.gpu) for key in lambda_lig_dict}
        sampled_y_lig_dict = {key: sampled_y_lig_dict[key].cuda(args.gpu) for key in sampled_y_lig_dict}
        P_rec_dict = {key: P_rec_dict[key].cuda(args.gpu) for key in P_rec_dict}
        lambda_rec_dict = {key: lambda_rec_dict[key].cuda(args.gpu) for key in lambda_rec_dict}
        sampled_y_rec_dict = {key: sampled_y_rec_dict[key].cuda(args.gpu) for key in sampled_y_rec_dict}


    # save ori_data_list
    ori_data_list = copy.deepcopy(data_list)

    # if type(ori_data_list) is not list:
    #     ori_data_list = ori_data_list.to("cpu").to_data_list()


    ori_meter = evaluate_all_ori_rmsds(data_list)
    _,_,_,_,_ = ori_meter.summarize(verbose=True)

    names = ori_meter.names


    # array = np.array(ori_meter.complex_rmsd_list)

    # # Get the indices of the smallest 10 elements
    # sml_indices = np.argsort(array)[:5]
    # lrg_indices = np.argsort(array)[-5:]

    # print("sml_indices", sml_indices, "names", [names[i] for i in sml_indices])
    # print("lrg_indices", lrg_indices, "names", [names[i] for i in lrg_indices])

    # # construct a dict 
    # dict = {"sml_indices": {}, "lrg_indices": {}}
    # for i in sml_indices:
    #     dict["sml_indices"][i] = {'name': names[i], 'complex_rmsd': array[i]}
    # for i in lrg_indices:
    #     dict["lrg_indices"][i] = {'name': names[i], 'complex_rmsd': array[i]}
    
    # # save the dict
    # with open("deform_NM_100_sml_lrg_complexes.pkl", 'wb') as f:
    #     pickle.dump(dict, f)

    visualize_first_n_samples = min(visualize_first_n_samples, len(data_list))
    the_visualization_dir = create_visualization_directories(visualization_dir, epoch)

    print('the_visualization_dir', the_visualization_dir)
    

    for i in range(visualize_first_n_samples):
        write_pdb_each(f"{the_visualization_dir}/{names[i].split('/')[1]}.pdb", ori_data_list[i]["receptor"].bb_atoms, ori_data_list[i].rec_res_name_b, ori_data_list[i]["ligand"].bb_atoms, ori_data_list[i].lig_res_name_b, rwmode='w')





    # # For visualization
    # for i in range(visualize_first_n_samples):
    #     write_pdb(visualization_values[i], graph_gts[i], "receptor",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-receptor.pdb")
    #     write_pdb(visualization_values[i], graph_gts[i], "ligand",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-gt.pdb")
    #     write_pdb(visualization_values[i], data_list[i], "ligand",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-0.pdb")

    
        
    cat_updates = {}

    # the last active mode index for each ligand and receptor
    ind_active_modes_lig = {key: -1 for key in P_lig_dict}
    ind_active_modes_rec = {key: -1 for key in P_rec_dict}

    # sample
    for t_idx in range(args.num_steps):
        
        # same time for all diffusion variables
        tr_schedule, rot_schedule, flex_rec_schedule, flex_lig_schedule = timesteps, timesteps, timesteps, timesteps

        t_flex_rec, t_flex_lig, t_rot, t_tr = flex_rec_schedule[t_idx], flex_lig_schedule[t_idx], rot_schedule[t_idx], tr_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < args.num_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < args.num_steps - 1 else rot_schedule[t_idx]
        dt_flex_rec = flex_rec_schedule[t_idx] - flex_rec_schedule[t_idx + 1] if t_idx < args.num_steps - 1 else flex_rec_schedule[t_idx]
        dt_flex_lig = flex_lig_schedule[t_idx] - flex_lig_schedule[t_idx + 1] if t_idx < args.num_steps - 1 else flex_lig_schedule[t_idx]

        # create new loader with current step graphs
        if torch.cuda.is_available() and args.num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader

        test_loader = loader(data_list, batch_size=args.batch_size)
        new_data_list = []  # updated every step


        # print("!!!!!!!! rank: ", args.rank, "test_loader.data: ", test_loader.data, "!!!!!!!!")

        for com_idx, complex_graphs in enumerate(test_loader):

            
            # move to CUDA
            #complex_graphs = complex_graphs.cuda()
            if torch.cuda.is_available() and args.num_gpu == 1:
                complex_graphs = complex_graphs.cuda(args.gpu)

            # this MAY differ from args.batch_size
            # based on # GPUs and last batch
            if type(complex_graphs) is list:
                batch_size = len(complex_graphs)
            else:
                batch_size = complex_graphs.num_graphs

            # convert to sigma space and save time
            flex_tr_rec_s, flex_tr_lig_s, flex_rec_s, flex_lig_s, rot_s, tr_s = transform.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr)
            if t_idx < args.num_steps - 1:
                flex_tr_rec_s_1, flex_tr_lig_s_1, _, _, _, _ = transform.noise_schedule(flex_rec_schedule[t_idx + 1], flex_lig_schedule[t_idx + 1], t_rot, t_tr)
            else:
                flex_tr_rec_s_1, flex_tr_lig_s_1 = 0, 0

            ds_flex_tr_rec_s = flex_tr_rec_s - flex_tr_rec_s_1
            ds_flex_tr_lig_s = flex_tr_lig_s - flex_tr_lig_s_1

            # to data list
            if type(complex_graphs) is not list:
                complex_graphs = complex_graphs.to_data_list()

            for i, complex_graph in enumerate(complex_graphs):
                activate_mode_cutoff = 5.0
                name = complex_graph.name

                # the newly activated modes last index (P_lig_dict, lambda_lig_dict, sampled_y_lig_dict, P_rec_dict, lambda_rec_dict, sampled_y_rec_dict)
                last_ind_lig = (lambda_lig_dict[name] * flex_tr_lig_s < activate_mode_cutoff).sum() - 1
                last_ind_rec = (lambda_rec_dict[name] * flex_tr_rec_s < activate_mode_cutoff).sum() - 1
                if last_ind_lig > ind_active_modes_lig[name]: # [????: X_u or the first few steps with at least one active mode]
                    # ind_active_modes_lig[name]+1:last_ind_lig+1
                    # P_new_active = P_lig_dict[name][:, (ind_active_modes_lig[name]+1):(last_ind_lig+1)]
                    # y_sampled_new_active = sampled_y_lig_dict[name][(ind_active_modes_lig[name]+1):(last_ind_lig+1)]
                    # ca_lig_tr = (P_new_active@y_sampled_new_active).reshape((-1, 3)) # (n,3)

                    # # print average ca_lig_tr norm
                    # print("average ca_lig_tr norm", torch.norm(ca_lig_tr, dim=1).mean())

                    # # update CA
                    # complex_graph["ligand"].pos = complex_graph["ligand"].pos + ca_lig_tr
                    # # update N, CA, C
                    # complex_graph["ligand"].bb_atoms = complex_graph["ligand"].bb_atoms + ca_lig_tr[:,None,:]

                    # update active modes
                    ind_active_modes_lig[name] = last_ind_lig
                elif last_ind_lig == -1:
                    ind_active_modes_lig[name] = -1
                else:
                    pass

                if last_ind_rec > ind_active_modes_rec[name]:
                    # P_new_active = P_rec_dict[name][:, (ind_active_modes_rec[name]+1):(last_ind_rec+1)]
                    # y_sampled_new_active = sampled_y_rec_dict[name][(ind_active_modes_rec[name]+1):(last_ind_rec+1)]
                    # ca_rec_tr = (P_new_active@y_sampled_new_active).reshape((-1, 3))

                    # complex_graph["receptor"].pos = complex_graph["receptor"].pos + ca_rec_tr
                    # complex_graph["receptor"].bb_atoms = complex_graph["receptor"].bb_atoms + ca_rec_tr[:,None,:]

                    ind_active_modes_rec[name] = last_ind_rec
                elif last_ind_rec == -1:
                    ind_active_modes_rec[name] = -1
                else:
                    pass

            if type(complex_graphs) is list:
                complex_graphs = Batch.from_data_list(complex_graphs)

            device_for_set_time = complex_graphs["ligand"]["pos"].device if torch.cuda.is_available() else None
            if type(complex_graphs) is list:
                for g in complex_graphs:
                    set_time_flexSDE_Rt(g, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device_for_set_time)
            else:
                set_time_flexSDE_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr, batch_size, device_for_set_time)
            
            with torch.no_grad():
                outputs = model(complex_graphs)

            coord_r_score = outputs["coord_r_pred"]
            coord_l_score = outputs["coord_l_pred"]
            lddt_score = outputs["lddt_pred"]
            rot_score = outputs["rot_pred"]
            tr_score = outputs["tr_pred"]

            if args.res_rot_flex:
                rec_res_rot_score = outputs["rec_res_rot_pred"]
                lig_res_rot_score = outputs["lig_res_rot_pred"]

            # print("lddt", lddt_score.reshape((batch_size, -1)))

            tr_g = tr_s * torch.sqrt(torch.tensor(2 * np.log(args.tr_s_max / args.tr_s_min)))
            tr_f = (tr_g/tr_s) ** 2 * dt_tr
            rot_g = 2 * rot_s * torch.sqrt(torch.tensor(np.log(args.rot_s_max / args.rot_s_min)))
            rot_f = dt_rot * (rot_g/rot_s) ** 2

            if args.ode:
                # tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()#
                # rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()#

                tr_perturb = torch.clamp(tr_score.cpu(), min=-args.tr_s_max, max=args.tr_s_max)# torch.clamp(tr_score, min=-20, max=20)#(inference_steps-t_idx) #* model_args.tr_sigma_max / inference_steps #+ torch.normal(mean=0, std=tr_sigma, size=(b, 3))  / (1+t_idx) #(inference_steps-t_idx)#
                rot_perturb = rot_score.cpu()#+ torch.normal(mean=0, std=1, size=(b, 3)) / (1+t_idx)#
            else: # args.no_final_noise defaulted as False
                tr_z = torch.zeros((batch_size, 3)) if args.no_random or (args.no_final_noise and t_idx == args.num_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(batch_size, 3))
                # tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()
                tr_perturb = torch.clamp(tr_score.cpu()+tr_g*np.sqrt(dt_tr)*tr_z, min=-args.tr_s_max, max=args.tr_s_max)
                rot_z = torch.zeros((batch_size, 3)) if args.no_random or (args.no_final_noise and t_idx == args.num_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(batch_size, 3))
                # rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()
                rot_perturb = rot_score.cpu() + rot_g * np.sqrt(dt_rot) * rot_z

            n_rec = complex_graphs['receptor'].pos.shape[0]
            n_lig = complex_graphs['ligand'].pos.shape[0]

            if args.ode or 1:
                protein_dynamic = True
                # if tr_s < args.zeroing_cutoff and protein_dynamic: #!!!!!!

                if protein_dynamic: #tr_s < 15.0 and protein_dynamic: #!!!!!!

                    ut=1-np.exp(-args.c*t_flex_rec)

                    try:
                        sampling_factor_dict = {20:27.0, 10:11.0, 5:5.5}
                        sampling_factor = sampling_factor_dict[int(args.c)]
                    except:
                        raise Exception("sampling_factor not found")

                    # flex_rec_perturb = coord_r_score.cpu() *27.0 * (1/ut - 1)/ (args.num_steps)
                    # flex_lig_perturb = coord_l_score.cpu() *27.0 * (1/ut - 1)/ (args.num_steps)


                    # print("here!!!!!!!")
                    # print(transform.noise_schedule.A_factor, transform.noise_schedule.B_exp)
                    # print("here!!!!!!")
                    if args.res_rot_flex:
                        rec_res_rot_perturb = rec_res_rot_score.cpu() *sampling_factor * (1/ut - 1)/ (args.num_steps)
                        lig_res_rot_perturb = lig_res_rot_score.cpu() *sampling_factor * (1/ut - 1)/ (args.num_steps)
                    # res_chi_perturb = res_chi_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                    # res_tr_perturb = res_tr_score.cpu() / (t_idx+inference_steps*0.1)
                    # res_rot_perturb = res_rot_score.cpu() / (t_idx+inference_steps*0.1)
                    # res_chi_perturb = res_chi_score.cpu() / (t_idx+inference_steps*0.1)



                elif False and tr_s < 6.0 and protein_dynamic:

                    if transform.noise_schedule.A_factor == 5.0 and transform.noise_schedule.B_exp == 0.3:
                        flex_rec_perturb = coord_r_score.cpu() / (args.num_steps-t_idx+args.num_steps*0.25)
                        flex_lig_perturb = coord_l_score.cpu() / (args.num_steps-t_idx+args.num_steps*0.25)
                        # print("here!!!!!!!")
                        # print(transform.noise_schedule.A_factor, transform.noise_schedule.B_exp)
                        # print("here!!!!!!")
                        if args.res_rot_flex:
                            rec_res_rot_perturb = rec_res_rot_score.cpu() / (args.num_steps-t_idx+args.num_steps*0.25)
                            lig_res_rot_perturb = lig_res_rot_score.cpu() / (args.num_steps-t_idx+args.num_steps*0.25)
                        # res_chi_perturb = res_chi_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                        # res_tr_perturb = res_tr_score.cpu() / (t_idx+inference_steps*0.1)
                        # res_rot_perturb = res_rot_score.cpu() / (t_idx+inference_steps*0.1)
                        # res_chi_perturb = res_chi_score.cpu() / (t_idx+inference_steps*0.1)
                    else:
                        flex_rec_perturb = coord_r_score.cpu() / (args.num_steps-t_idx)*args.B_exp
                        flex_lig_perturb = coord_l_score.cpu() / (args.num_steps-t_idx)*args.B_exp
                        print("here??????")
                        print(transform.noise_schedule.A_factor, transform.noise_schedule.B_exp)
                        print("here??????")
                else:
                    # flex_rec_perturb = torch.zeros((n_rec, 3))
                    # flex_lig_perturb = torch.zeros((n_lig, 3))

                    if args.res_rot_flex:
                        rec_res_rot_perturb = torch.zeros((n_rec, 3))
                        lig_res_rot_perturb = torch.zeros((n_lig, 3))
                    # res_chi_perturb = torch.zeros((n, 5))
                # if t_idx <= inference_steps - 1:
                #     res_tr_perturb = res_tr_score.cpu() / inference_steps * 10#(0.5 * res_tr_g ** 2 * dt_res_tr * res_tr_score.cpu()).cpu()
                #     res_rot_perturb = res_rot_score.cpu() / inference_steps * 10#(0.5 * res_rot_score.cpu() * dt_res_rot * res_rot_g ** 2).cpu()
                # else:
                # res_tr_perturb = torch.zeros((n, 3))
                # res_rot_perturb = torch.zeros((n, 3))
                # res_chi_perturb = torch.zeros((n, 5))
            else:
                # # res_tr_g = 3*torch.sqrt(torch.tensor(2 * np.log(model_args.res_tr_sigma_max / model_args.res_tr_sigma_min)))
                # # res_rot_g = 3*torch.sqrt(torch.tensor(2 * np.log(model_args.res_rot_sigma_max / model_args.res_rot_sigma_min)))

                # # res_tr_z = torch.zeros((n, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                # #     else torch.normal(mean=0, std=model_args.res_tr_sigma_max/20, size=(n, 3))
                # # res_tr_perturb = (res_tr_g ** 2 * dt_res_tr * res_tr_score.cpu() + res_tr_g * np.sqrt(dt_res_tr) * res_tr_z).cpu()
                # #
                # # res_rot_z = torch.zeros((n, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                # #     else torch.normal(mean=0, std=model_args.res_rot_sigma_max/20, size=(n, 3))
                # # res_rot_perturb = (res_rot_score.cpu() * dt_res_rot * res_rot_g ** 2 + res_rot_g * np.sqrt(dt_res_rot) * res_rot_z).cpu()
                # res_tr_perturb = torch.zeros((n, 3))
                # res_rot_perturb = torch.zeros((n, 3))
                # res_chi_z = torch.zeros((n, 5)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                #     else torch.normal(mean=0, std=1, size=(n, 5))
                # res_chi_perturb = res_chi_score.cpu() + res_chi_z*dt_res_chi
                
                flex_rec_perturb = torch.zeros((n_rec, 3))
                flex_lig_perturb = torch.zeros((n_lig, 3))

            # if DEBUG:
            #     print("Before clamping, rec max perturbation:", flex_rec_perturb.max().item())
            #     print("Before clamping, lig max perturbation:", flex_lig_perturb.max().item())



            

            # to data list
            if type(complex_graphs) is not list:
                complex_graphs = complex_graphs.to("cpu").to_data_list()



            # apply transformations
            rec_i = 0
            lig_i = 0
            for i, complex_graph in enumerate(complex_graphs):
                name = complex_graph.name

                print("!!!!", name, "!!!!")

                # # rec res-level tr perturb   
                # delta_ca_rec = complex_graph["receptor"].pos - complex_graph["receptor"].pos_u_Rt # (n,3)
                # delta_ca_rec_reshaped = delta_ca_rec.reshape((-1,)) # (3n,)
                # P_rec = P_rec_dict[name]
                # H_rec = P_rec @ torch.diag(lambda_rec_dict[name]) @ P_rec.T # (3n,3n)
                # H_rec = H_rec.cpu()
                # P_rec = P_rec.cpu()
                # drift = ((H_rec@delta_ca_rec_reshaped/2 + P_rec @ P_rec.T @(coord_r_score.cpu()[rec_i:rec_i+complex_graph['receptor'].pos.shape[0]].reshape(-1,))).reshape((-1,3))) * ds_flex_tr_rec_s
                # flex_rec_perturb = drift + torch.sqrt(torch.tensor(ds_flex_tr_rec_s)) * (P_rec @ torch.normal(mean=0, std=1, size=(P_rec.shape[1],))).reshape((-1,3))
                # print("flex_rec_perturb average", torch.mean(flex_rec_perturb, dim=0))
                # print("coord_r_score norm average", torch.norm(coord_r_score.cpu()[rec_i:rec_i+complex_graph['receptor'].pos.shape[0]], dim=1).mean())
                # print("torch.sqrt(torch.tensor(ds_flex_tr_rec_s))", torch.sqrt(torch.tensor(ds_flex_tr_rec_s)))

                # # lig res-level tr perturb
                # delta_ca_lig = complex_graph["ligand"].pos - complex_graph["ligand"].pos_u_Rt
                # print("complex_graph['ligand'].pos", complex_graph["ligand"].pos.mean())
                # print("complex_graph['ligand'].pos_u_Rt", complex_graph["ligand"].pos_u_Rt.mean())

                # delta_ca_lig_reshaped = delta_ca_lig.reshape((-1,))
                # P_lig = P_lig_dict[name]
                # print("P_lig", P_lig.mean())
                # H_lig = P_lig @ torch.diag(lambda_lig_dict[name]) @ P_lig.T
                # H_lig = H_lig.cpu()
                # P_lig = P_lig.cpu()
                # print("H_lig", H_lig.mean())
                # print("delta_ca_lig_reshaped", delta_ca_lig_reshaped.mean())
                # drift = ((H_lig@delta_ca_lig_reshaped/2 + P_lig @ P_lig.T @(coord_l_score.cpu()[lig_i:lig_i+complex_graph['ligand'].pos.shape[0]].reshape(-1,))).reshape((-1,3))) * ds_flex_tr_lig_s
                # print("drift mean",torch.mean(drift, dim=0))
                # flex_lig_perturb = drift + torch.sqrt(torch.tensor(ds_flex_tr_lig_s)) * (P_lig @ torch.normal(mean=0, std=1, size=(P_lig.shape[1],))).reshape((-1,3))
                # print("flex_lig_perturb average", torch.mean(flex_lig_perturb, dim=0))
                # print("coord_l_score norm average", torch.norm(coord_l_score.cpu()[lig_i:lig_i+complex_graph['ligand'].pos.shape[0]], dim=1).mean())
                # print("torch.sqrt(torch.tensor(ds_flex_tr_lig_s))", torch.sqrt(torch.tensor(ds_flex_tr_lig_s)))

                # # update P_lig according to the R and t
                # rot_update = rot_perturb[i:i + 1].squeeze(0)
                # rot_mat = axis_angle_to_matrix(rot_update.squeeze())

                # if torch.cuda.is_available() and args.num_gpu == 1:
                #     rot_mat = rot_mat.cuda(args.gpu)

                # print("rot_mat", rot_mat)


                # P_lig = P_lig_dict[name]
                # _, m = P_lig.shape
                # P_lig_dict[name] = ((P_lig.reshape((-1,3,m)).permute((0,2,1))) @ (rot_mat.T[None,:,:])).permute((0,2,1)).reshape((-1,m))

                # flex_rec_perturb = torch.clamp(flex_rec_perturb, min=-20, max=20)       # safe perturb
                # flex_lig_perturb = torch.clamp(flex_lig_perturb, min=-20, max=20)       # safe perturb


                # # print average flex_rec_perturb norm
                # print("average flex_rec_perturb norm", torch.norm(flex_rec_perturb, dim=1).mean())
                # print("average flex_lig_perturb norm", torch.norm(flex_lig_perturb, dim=1).mean())

                # # !!!!!!
                # flex_lig_perturb = torch.zeros((complex_graph['ligand'].pos.shape[0], 3))
                # flex_rec_perturb = torch.zeros((complex_graph['receptor'].pos.shape[0], 3))

            

                # print("ind_active_modes_rec[name]", ind_active_modes_rec[name])
                # print("ind_active_modes_lig[name]", ind_active_modes_lig[name])

                # rec res-level tr perturb
                if ind_active_modes_rec[name] == -1:
                    flex_rec_perturb = torch.zeros((complex_graph['receptor'].pos.shape[0], 3))
                else:
                    delta_ca_rec = complex_graph["receptor"].pos - complex_graph["receptor"].pos_u_Rt # (n,3)
                    delta_ca_rec_reshaped = delta_ca_rec.reshape((-1,)) # (3n,)
                    P_rec = P_rec_dict[name][:, :(ind_active_modes_rec[name]+1)] # (3n, m)
                    H_rec = P_rec @ torch.diag(lambda_rec_dict[name][:(ind_active_modes_rec[name]+1)]) @ P_rec.T # (3n,3n)
                    H_rec = H_rec.cpu()
                    P_rec = P_rec.cpu()
                    Lambda_rec = (lambda_rec_dict[name][:(ind_active_modes_rec[name]+1)]).cpu()
                    if args.balanced_diffusion:
                        drift = ((H_rec@delta_ca_rec_reshaped/2 + H_rec @(coord_r_score.cpu()[rec_i:rec_i+complex_graph['receptor'].pos.shape[0]].reshape(-1,))).reshape((-1,3))) * ds_flex_tr_rec_s


                        # print("drift.shape:", drift.shape)
                        # print("ds_flex_tr_rec_s:", ds_flex_tr_rec_s)
                        # print("P_rec.shape:", P_rec.shape)
                        # print("Lambda_rec.shape:", Lambda_rec.shape)
                        # temp = torch.sqrt(Lambda_rec) * torch.normal(mean=0, std=1, size=(P_rec.shape[1], ))
                        # print("temp.shape:", temp.shape)
                        # prod = P_rec @ temp
                        # print("prod.shape:", prod.shape)
                        # prod_reshaped = prod.reshape((-1, 3))
                        # print("prod_reshaped.shape:", prod_reshaped.shape)

                        flex_rec_perturb = drift + torch.sqrt(torch.tensor(ds_flex_tr_rec_s)) * (P_rec @ (torch.sqrt(Lambda_rec) *torch.normal(mean=0, std=1, size=(P_rec.shape[1],)))).reshape((-1,3))
                    else:
                        if args.neg_drift:
                            drift = ((-H_rec@delta_ca_rec_reshaped/2 + P_rec @ P_rec.T @(coord_r_score.cpu()[rec_i:rec_i+complex_graph['receptor'].pos.shape[0]].reshape(-1,))).reshape((-1,3))) * ds_flex_tr_rec_s
                        else:
                            drift = ((H_rec@delta_ca_rec_reshaped/2 + P_rec @ P_rec.T @(coord_r_score.cpu()[rec_i:rec_i+complex_graph['receptor'].pos.shape[0]].reshape(-1,))).reshape((-1,3))) * ds_flex_tr_rec_s
                        flex_rec_perturb = drift + torch.sqrt(torch.tensor(ds_flex_tr_rec_s)) * (P_rec @ torch.normal(mean=0, std=1, size=(P_rec.shape[1],))).reshape((-1,3))
                

                # lig res-level tr perturb
                if ind_active_modes_lig[name] == -1:
                    flex_lig_perturb = torch.zeros((complex_graph['ligand'].pos.shape[0], 3))
                else:
                    delta_ca_lig = complex_graph["ligand"].pos - complex_graph["ligand"].pos_u_Rt
                    print("complex_graph['ligand'].pos", complex_graph["ligand"].pos.mean())
                    print("complex_graph['ligand'].pos_u_Rt", complex_graph["ligand"].pos_u_Rt.mean())

                    delta_ca_lig_reshaped = delta_ca_lig.reshape((-1,))
                    P_lig = P_lig_dict[name][:, :(ind_active_modes_lig[name]+1)]
                    print("P_lig", P_lig.mean())
                    H_lig = P_lig @ torch.diag(lambda_lig_dict[name][:(ind_active_modes_lig[name]+1)]) @ P_lig.T
                    H_lig = H_lig.cpu()
                    P_lig = P_lig.cpu()
                    Lambda_lig = (lambda_lig_dict[name][:(ind_active_modes_lig[name]+1)]).cpu()
                    print("H_lig", H_lig.mean())
                    print("delta_ca_lig_reshaped", delta_ca_lig_reshaped.mean())

                    if args.balanced_diffusion:
                        drift = ((H_lig@delta_ca_lig_reshaped/2 + H_lig @(coord_l_score.cpu()[lig_i:lig_i+complex_graph['ligand'].pos.shape[0]].reshape(-1,))).reshape((-1,3))) * ds_flex_tr_lig_s
                        flex_lig_perturb = drift + torch.sqrt(torch.tensor(ds_flex_tr_lig_s)) * (P_lig @ (torch.sqrt(Lambda_lig) *torch.normal(mean=0, std=1, size=(P_lig.shape[1],)))).reshape((-1,3))
                    else:
                        if args.neg_drift:
                            drift = ((-H_lig@delta_ca_lig_reshaped/2 + P_lig @ P_lig.T @(coord_l_score.cpu()[lig_i:lig_i+complex_graph['ligand'].pos.shape[0]].reshape(-1,))).reshape((-1,3))) * ds_flex_tr_lig_s
                        else:
                            drift = ((H_lig@delta_ca_lig_reshaped/2 + P_lig @ P_lig.T @(coord_l_score.cpu()[lig_i:lig_i+complex_graph['ligand'].pos.shape[0]].reshape(-1,))).reshape((-1,3))) * ds_flex_tr_lig_s
                            
                        flex_lig_perturb = drift + torch.sqrt(torch.tensor(ds_flex_tr_lig_s)) * (P_lig @ torch.normal(mean=0, std=1, size=(P_lig.shape[1],))).reshape((-1,3))
                    print("coord_l_score norm average", torch.norm(coord_l_score.cpu()[lig_i:lig_i+complex_graph['ligand'].pos.shape[0]], dim=1).mean())
                    print("torch.sqrt(torch.tensor(ds_flex_tr_lig_s))", torch.sqrt(torch.tensor(ds_flex_tr_lig_s)))

                    # update P_lig according to the R and t
                    rot_update = rot_perturb[i:i + 1].squeeze(0)
                    rot_mat = axis_angle_to_matrix(rot_update.squeeze())

                    if torch.cuda.is_available() and args.num_gpu == 1:
                        rot_mat = rot_mat.cuda(args.gpu)

                    print("rot_mat", rot_mat)

                    

                    P_lig = P_lig_dict[name]
                    _, m = P_lig.shape
                    P_lig_dict[name] = ((P_lig.reshape((-1,3,m)).permute((0,2,1))) @ (rot_mat.T[None,:,:])).permute((0,2,1)).reshape((-1,m))

                flex_rec_perturb = torch.clamp(flex_rec_perturb, min=-20, max=20)       # safe perturb
                flex_lig_perturb = torch.clamp(flex_lig_perturb, min=-20, max=20)       # safe perturb


                # # print average flex_rec_perturb norm
                # print("average flex_rec_perturb norm", torch.norm(flex_rec_perturb, dim=1).mean())
                # print("average flex_lig_perturb norm", torch.norm(flex_lig_perturb, dim=1).mean())


                new_data_list.extend([transform.apply_updates(complex_graph, tr_update=tr_perturb[i:i + 1], rot_update=rot_perturb[i:i + 1].squeeze(0),
                                    flex_rec_update=flex_rec_perturb, 
                                    flex_lig_update=flex_lig_perturb, 
                                    flex_rec_rot_update=rec_res_rot_perturb[rec_i:rec_i+complex_graph['receptor'].pos.shape[0]] if args.res_rot_flex else None,
                                    flex_lig_rot_update=lig_res_rot_perturb[lig_i:lig_i+complex_graph['ligand'].pos.shape[0]] if args.res_rot_flex else None)])
                rec_i += complex_graph['receptor'].pos.shape[0]
                lig_i += complex_graph['ligand'].pos.shape[0]

            # === end of batch ===
            #printt(f'finished batch {com_idx}')

        # for i in range(visualize_first_n_samples):
        #     write_pdb(visualization_values[i], new_data_list[i], "ligand",
        #               f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-{t_idx + 1}.pdb")

        # update starting point for next step
        assert len(new_data_list) == len(data_list)
        data_list = new_data_list

        for i in range(visualize_first_n_samples):
            write_pdb_each(f"{the_visualization_dir}/{names[i].split('/')[1]}.pdb", data_list[i]["receptor"].bb_atoms, data_list[i].rec_res_name_b, data_list[i]["ligand"].bb_atoms, data_list[i].lig_res_name_b, rwmode='a')


        # updates_meter = evaluate_all_rmsds(new_data_list)
        # # _,_,_,_ = updates_meter.summarize(verbose=True) #[????]
        # complex_rmsd_list, interface_rmsd_list, names = updates_meter.complex_rmsd_list, updates_meter.interface_rmsd_list, updates_meter.names

        printt(f"Completed {t_idx} out of {args.num_steps} steps")

    # all_lddt_pred = []
    # all_affinity_pred = []
    # loader = DataLoader(data_list, batch_size=batch_size)
    # for complex_graph_batch in loader:
    #     b = complex_graph_batch.num_graphs
    #     t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi = [0.6] * 6
    #     complex_graph_batch = complex_graph_batch.to(device)
    #     set_time(complex_graph_batch, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, b, model_args.all_atoms, device)
    #     with torch.no_grad():
    #         lddt_pred, affinity_pred, tr_score, rot_score, tor_score, res_tr_score, res_rot_score, res_chi_score = model(complex_graph_batch)
    #     all_lddt_pred.append(lddt_pred)
    #     all_affinity_pred.append(affinity_pred)
    # all_lddt_pred = torch.cat(all_lddt_pred,dim=0)
    # all_affinity_pred = torch.cat(all_affinity_pred,dim=0)
    # # all_lddt_pred, all_affinity_pred = pred_lddt_and_affinity(data_list, model, batch_size, device, model_args)

    return data_list#, batch_size_to_return


def write_pdb_each(file_name, bb_coords_chain_R, sequence_chain_R, bb_coords_chain_L, sequence_chain_L, rwmode=None):
    """
    Write backbone coordinates for two chains into a PDB file.

    Parameters:
        file_name (str): Name of the PDB file to write.
        bb_coords_chain_R (numpy.ndarray): Backbone coordinates for chain R with shape (n_res, 3, 3).
        sequence_chain_R (list): List of three-letter amino acid codes for chain R with length n_res.
        bb_coords_chain_L (numpy.ndarray): Backbone coordinates for chain L with shape (n_res, 3, 3).
        sequence_chain_L (list): List of three-letter amino acid codes for chain L with length n_res.
    """
    # Assertions to ensure the inputs are valid
    assert bb_coords_chain_R.shape[0] == len(sequence_chain_R), "Mismatch between residues and sequence length in Chain R."
    assert bb_coords_chain_R.shape[1] == 3 and bb_coords_chain_R.shape[2] == 3, "Invalid shape for bb_coords_chain_R. Expected (n_res, 3, 3)."
    assert bb_coords_chain_L.shape[0] == len(sequence_chain_L), "Mismatch between residues and sequence length in Chain L."
    assert bb_coords_chain_L.shape[1] == 3 and bb_coords_chain_L.shape[2] == 3, "Invalid shape for bb_coords_chain_L. Expected (n_res, 3, 3)."
    
    if rwmode is None:
        if os.path.exists(file_name):
            rwmode = 'a'
        else:
            rwmode = 'w'

    with open(file_name, rwmode) as pdb_file:
        atom_counter = 1

        # Write Chain R
        residue_counter = 1
        for res_idx, (res_coords, res_name) in enumerate(zip(bb_coords_chain_R, sequence_chain_R)):
            atom_names = ['N', 'CA', 'C']
            for atom_idx, atom_name in enumerate(atom_names):
                x, y, z = res_coords[atom_idx]
                pdb_file.write(
                    f"ATOM  {atom_counter:5d} {atom_name:^4s} {res_name:<3s} R{residue_counter:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                )
                atom_counter += 1
            residue_counter += 1

        # Write Chain L
        residue_counter = 1
        for res_idx, (res_coords, res_name) in enumerate(zip(bb_coords_chain_L, sequence_chain_L)):
            atom_names = ['N', 'CA', 'C']
            for atom_idx, atom_name in enumerate(atom_names):
                x, y, z = res_coords[atom_idx]
                pdb_file.write(
                    f"ATOM  {atom_counter:5d} {atom_name:^4s} {res_name:<3s} L{residue_counter:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                )
                atom_counter += 1
            residue_counter += 1

        # Add END record to indicate the end of the PDB file
        pdb_file.write("ENDMDL\n")



def create_visualization_directories(top_visualization_dir, epoch, pdb_names=None):
    if pdb_names is None:
        directory = f"{top_visualization_dir}/epoch-{epoch}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
    
    visualization_dirs = [f"{top_visualization_dir}/epoch-{epoch}/{pdb_name}" for pdb_name in pdb_names]
    for directory in visualization_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return visualization_dirs

def get_four_letters_pdb_identifier(pdb_name):
    return pdb_name.split('/')[-1].split('.')[0]


def write_pdb(item, graph, part, path):
    lines = to_pdb_lines(item, graph, part)
    with open(path, "w") as file:
        file.writelines(lines)


def to_pdb_lines(visualization_values, graph, part):
    assert part in ("ligand", "receptor", "both"), "Part should be ligand or receptor"
    parts = ["ligand", "receptor"] if part == "both" else [part]

    lines = []
    for part in parts:
        this_vis_values = visualization_values[part]
        this_vis_values = {k: v.strip() if type(v) is str else v for k, v in this_vis_values.items()}
        for i, resname in enumerate(this_vis_values["resname"]):
            xyz = graph[part].pos[i]

            line = f'ATOM  {i + 1:>5} {this_vis_values["atom_name"][i]:>4} '
            line = line + f'{resname} {this_vis_values["chain"][i]}{this_vis_values["residue"][i]:>4}    '.replace("<Chain id=", "").replace(">", "")
            line = line + f'{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}'
            line = line + '  1.00  0.00          '
            line = line + f'{this_vis_values["element"][i]:>2} 0\n'
            lines.append(line)

    return lines

# [nm]
def get_timesteps(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1] # endings inclusive # num of evenly spaced values






# [nm]
def randomize_position_AND_use_unbound(data_list, args, ESP=1e-6, C=1.0, factor=0.1):
    """
        Modify COPY of data_list objects
    """

    # data_list = copy.deepcopy(data_list)


    data_list = data_list.deepcopy()

    P_lig_dict = {}
    P_rec_dict = {}

    lambda_lig_dict = {}
    lambda_rec_dict = {}

    sampled_y_lig_dict = {}
    sampled_y_rec_dict = {}

    

    
    for i, complex_graph in enumerate(data_list):

        name = complex_graph.name

        P_lig = complex_graph["ligand"].P
        P_rec = complex_graph["receptor"].P

        lambda_lig = complex_graph["ligand"].Lambda
        lambda_rec = complex_graph["receptor"].Lambda

        # non_zero_ind = lambda_lig > ESP
        # lambda_lig = lambda_lig[non_zero_ind]
        # P_lig = P_lig[:, non_zero_ind]

        # non_zero_ind = lambda_rec > ESP
        # lambda_rec = lambda_rec[non_zero_ind]
        # P_rec = P_rec[:, non_zero_ind]

        
        lambda_lig = lambda_lig[6:]
        P_lig = P_lig[:, 6:]

        lambda_rec = lambda_rec[6:]
        P_rec = P_rec[:, 6:]
        if name == 'qs/2qsf.pdb1_0.dill':
            print(name)
            print("lambda_lig", lambda_lig)
            print("lambda_rec", lambda_rec)

            print("P_lig mean", P_lig.mean())
            print("P_rec mean", P_rec.mean())

        if args.eigenscale == 'log':
            # scale the eigenvalues by log(lambda_i/lambda_smallest) + C 
            print("use log scale")
            lambda_lig = torch.abs(lambda_lig)+ESP
            lambda_lig = torch.log(lambda_lig/lambda_lig[0]) + C

            lambda_rec = torch.abs(lambda_rec)+ESP
            lambda_rec = torch.log(lambda_rec/lambda_rec[0]) + C
        elif args.eigenscale == 'affine':
            print("use affine scale")
            lambda_lig = torch.abs(lambda_lig)+ESP
            lambda_lig = ((lambda_lig - lambda_lig.min())/(lambda_lig.max()-lambda_lig.min()) + C)*factor

            lambda_rec = torch.abs(lambda_rec)+ESP
            lambda_rec = ((lambda_rec - lambda_rec.min())/(lambda_rec.max()-lambda_rec.min()) + C)*factor
        else:
            raise ValueError("Invalid eigenscale")



        del complex_graph.H_rec
        del complex_graph.H_lig
        del complex_graph['receptor'].P
        del complex_graph['ligand'].P
        del complex_graph['receptor'].Lambda
        del complex_graph['ligand'].Lambda


        
        if args.balanced_diffusion:
            sampled_y_lig_dict[name] = torch.normal(torch.zeros_like(lambda_lig), torch.ones_like(lambda_lig))
            sampled_y_rec_dict[name] = torch.normal(torch.zeros_like(lambda_rec), torch.ones_like(lambda_rec))
        else:
            sampled_y_lig_dict[name] = torch.normal(torch.zeros_like(lambda_lig), torch.sqrt(1/lambda_lig))
            sampled_y_rec_dict[name] = torch.normal(torch.zeros_like(lambda_rec), torch.sqrt(1/lambda_rec))

        update_tr_lig = (P_lig@sampled_y_lig_dict[name]).reshape(-1,3) 
        update_tr_rec = (P_rec@sampled_y_rec_dict[name]).reshape(-1,3)

        # print("update_tr_lig mean", torch.mean(update_tr_lig, dim=0))
        # print("update_tr_rec mean", torch.mean(update_tr_rec, dim=0))



        flex_new_pos_l = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask]
        flex_new_pos_r = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask]

        print("mask len", len(complex_graph["ligand"].mask), sum(complex_graph["ligand"].mask))
        print("mask len", len(complex_graph["receptor"].mask), sum(complex_graph["receptor"].mask))
        
        
        complex_graph["ligand"].pos = flex_new_pos_l + update_tr_lig
        complex_graph["receptor"].pos = flex_new_pos_r + update_tr_rec

        if 'x_ori' in complex_graph["ligand"]:
            
            complex_graph["ligand"].x = complex_graph["ligand"].x_ori
            complex_graph["receptor"].x = complex_graph["receptor"].x_ori

            complex_graph["ligand", "contact", "ligand"].edge_index = complex_graph["ligand"].edge_index_b
            complex_graph["receptor", "contact", "receptor"].edge_index = complex_graph["receptor"].edge_index_b
        else:
            pass
        
        # lig random rotation
        pos = complex_graph["ligand"].pos
        center = torch.mean(pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix())
        pos = (pos - center) @ random_rotation.T.float()

        # lig random translation
        tr_update = torch.normal(0, args.tr_s_max, size=(1, 3))
        pos = pos + tr_update
        complex_graph["ligand"].pos = pos

        # ub receptor and rotated and translated ub ligand
        complex_graph["receptor"].pos_u_Rt = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask]

        pos_u = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask]
        center_u = torch.mean(pos_u, dim=0, keepdim=True)
        pos_u = (pos_u - center_u) @ random_rotation.T.float() + tr_update
        complex_graph["ligand"].pos_u_Rt = pos_u


        print('complex_graph["ligand"].pos_u_Rt', torch.mean(complex_graph["ligand"].pos_u_Rt, dim=0))
        print('complex_graph["ligand"].pos', torch.mean(complex_graph["ligand"].pos, dim=0))

        if args.res_rot_flex:
            # print(len(complex_graph["receptor"].mask), sum(complex_graph["receptor"].mask),'!!!')
            complex_graph['receptor'].bb_atoms = complex_graph['receptor'].bb_atoms_u[complex_graph["receptor"].mask]  + update_tr_rec[:,None,:]
            complex_graph['ligand'].bb_atoms = (complex_graph['ligand'].bb_atoms_u[complex_graph["ligand"].mask] - center[:,None,:]) @ ((random_rotation.T)[None,:,:]).float() + tr_update[:,None,:] + update_tr_lig[:,None,:]


        data_list.set_graph(i, complex_graph)

        _, m = P_lig.shape
        P_lig = ((P_lig.reshape((-1,3,m)).permute((0,2,1))) @ (random_rotation.T.float()[None,:,:])).permute((0,2,1)).reshape((-1,m))

        P_lig_dict[name] = P_lig
        lambda_lig_dict[name] = lambda_lig

        P_rec_dict[name] = P_rec
        lambda_rec_dict[name] = lambda_rec

        

    return data_list, P_lig_dict, lambda_lig_dict, sampled_y_lig_dict, P_rec_dict, lambda_rec_dict, sampled_y_rec_dict

