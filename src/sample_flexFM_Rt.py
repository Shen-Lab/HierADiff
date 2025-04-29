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
from scipy.spatial.transform import Rotation as R

from utils import printt
from geom_utils import set_time_flexFM_Rt, NoiseTransform_flexFM_Rt
from geom_utils.geometry import kabsch_torch

from evaluation.compute_rmsd_nm import evaluate_all_rmsds, evaluate_all_ori_rmsds


EPS = 1e-6


DEBUG_inv = False

DEBUG_steps_80 = False # True # 

DEBUG_no_random = True # False #



# [nm]
def sample_flexFM_Rt(data_list, model, args, epoch=0, visualize_first_n_samples=0,
           visualization_dir="./visualization", in_batch_size=None, output_coeffs=False, all_results=None, output_path=None, pkl_ind=None):
    """
        Run reverse process
    """
    DEBUG = False


    if in_batch_size is None:
        in_batch_size = args.batch_size
        
    # switch to eval mode
    model.eval()

    # stores various noise-related utils
    transform = NoiseTransform_flexFM_Rt(args)

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
    data_list = randomize_position_AND_use_unbound(data_list, args)

    # save ori_data_list
    ori_data_list = copy.deepcopy(data_list)

    # if type(ori_data_list) is not list:
    #     ori_data_list = ori_data_list.to("cpu").to_data_list()


    ori_meter = evaluate_all_ori_rmsds(data_list)
    _,_,_,_,_ = ori_meter.summarize(verbose=True)

    names = ori_meter.names

    if all_results is not None:
        if all_results == {}:
            # initalize all_results
            all_results = {'SEED':[]}
            for name in names:
                all_results[name] = {'IRMSD': [], 'CRMSD': [], 'pLDDT': [], 'tLDDT': [], 'LRMSD': [], 'fnat': [], 'gt_pred_recLig_bb_atoms': [], 'docking_quality': []}
        # put the current seed info
        all_results['SEED'].append(args.seed)


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
    
    # sampling results/lddt results dir
    the_output_dir = create_visualization_directories(output_path, epoch)
    print('the_output_dir', the_output_dir)





    # # For visualization
    # for i in range(visualize_first_n_samples):
    #     write_pdb(visualization_values[i], graph_gts[i], "receptor",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-receptor.pdb")
    #     write_pdb(visualization_values[i], graph_gts[i], "ligand",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-gt.pdb")
    #     write_pdb(visualization_values[i], data_list[i], "ligand",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-0.pdb")

    
        
    cat_updates = {}

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


        all_lddt_scores = {}


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
            if args.adaptive_schedule:
                flex_rec_s, flex_lig_s, rot_s, tr_s = transform.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr, adaptive_c=complex_graphs.adaptive_c)
            else:
                flex_rec_s, flex_lig_s, rot_s, tr_s = transform.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr)
            

            device_for_set_time = complex_graphs["ligand"]["pos"].device if torch.cuda.is_available() else None


            if args.adaptive_schedule:
                if type(complex_graphs) is list:
                    for g, every_flex_rec_s, every_flex_lig_s in zip(complex_graphs, flex_rec_s, flex_lig_s):
                        set_time_flexFM_Rt(g, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device_for_set_time, t_fm_deform_rec=every_flex_rec_s, t_fm_deform_lig=every_flex_lig_s)
                else:
                    set_time_flexFM_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr, batch_size, device_for_set_time, t_fm_deform_rec=flex_rec_s, t_fm_deform_lig=flex_lig_s)
            else:
                if type(complex_graphs) is list:
                    for g in complex_graphs:
                        set_time_flexFM_Rt(g, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device_for_set_time)
                else:
                    set_time_flexFM_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr, batch_size, device_for_set_time)
                

                


            # if type(complex_graphs) is list:
            #     for g in complex_graphs:
            #         set_time_flexFM_Rt(g, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device_for_set_time)
            # else:
            #     set_time_flexFM_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr, batch_size, device_for_set_time)
            

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

            for name in complex_graphs.name:
                all_lddt_scores[name] = lddt_score[complex_graphs.name.index(name)].cpu().numpy()
            

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

                if protein_dynamic and (args.adaptive_schedule): # and tr_s < args.zeroing_cutoff:
                    batch_rec = complex_graphs['receptor'].batch
                    batch_lig = complex_graphs['ligand'].batch
                    
                    # adaptive_c [bs, 1]
                    adaptive_c = complex_graphs.adaptive_c
                    # ut_r [n_rec_res, 1], ut_l [n_lig_res, 1]
                    ut_r = complex_graphs['receptor'].t_fm_deform[:, None].cpu()
                    ut_l = complex_graphs['ligand'].t_fm_deform[:, None].cpu()

                    # sampling_factor_r [n_rec_res, 1], sampling_factor_l [n_lig_res, 1]
                    sampling_factor_r = adaptive_c[batch_rec][:, None].cpu()
                    sampling_factor_l = adaptive_c[batch_lig][:, None].cpu()

                    # print("adaptive_c dtype", adaptive_c.dtype)
                    # print("ut_r dtype", ut_r.dtype)
                    # print("sampling_factor_r dtype", sampling_factor_r.dtype)
                    
                    # ut=1-np.exp(-adaptive_c*t_flex_rec)

                    # sampling_factor = adaptive_c

                    flex_rec_perturb = coord_r_score.cpu() *sampling_factor_r * (1/ut_r - 1)/ (args.num_steps)
                    flex_lig_perturb = coord_l_score.cpu() *sampling_factor_l * (1/ut_l - 1)/ (args.num_steps)

                    if args.res_rot_flex:   
                        rec_res_rot_perturb = rec_res_rot_score.cpu() *sampling_factor_r * (1/ut_r - 1)/ (args.num_steps)
                        lig_res_rot_perturb = lig_res_rot_score.cpu() *sampling_factor_l * (1/ut_l - 1)/ (args.num_steps)

                

                elif protein_dynamic and (not args.adaptive_schedule) and tr_s < args.zeroing_cutoff: #!!!!!!

                    ut=1-np.exp(-args.c*t_flex_rec)

                    try:
                        sampling_factor_dict = {20:27.0, 10:11.0, 5:5.5}
                        sampling_factor = sampling_factor_dict[int(args.c)]
                    except:
                        raise Exception("sampling_factor not found")


                    

                    flex_rec_perturb = coord_r_score.cpu() *sampling_factor * (1/ut - 1)/ (args.num_steps)
                    flex_lig_perturb = coord_l_score.cpu() *sampling_factor * (1/ut - 1)/ (args.num_steps)

                    # print("flex_rec_perturb", flex_rec_perturb)
                    # print("flex_lig_perturb", flex_lig_perturb)


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


                elif False and tr_s < 15.0 and protein_dynamic:

                    ut=1-np.exp(-20*t_flex_rec)

                    flex_rec_perturb = coord_r_score.cpu() *27.0 * (1/ut - 1)/ (args.num_steps)
                    flex_lig_perturb = coord_l_score.cpu() *27.0 * (1/ut - 1)/ (args.num_steps)
                    # print("here!!!!!!!")
                    # print(transform.noise_schedule.A_factor, transform.noise_schedule.B_exp)
                    # print("here!!!!!!")
                    if args.res_rot_flex:
                        rec_res_rot_perturb = rec_res_rot_score.cpu() *27.0 * (1/ut - 1)/ (args.num_steps)
                        lig_res_rot_perturb = lig_res_rot_score.cpu() *27.0 * (1/ut - 1)/ (args.num_steps)
                    # res_chi_perturb = res_chi_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                    # res_tr_perturb = res_tr_score.cpu() / (t_idx+inference_steps*0.1)
                    # res_rot_perturb = res_rot_score.cpu() / (t_idx+inference_steps*0.1)
                    # res_chi_perturb = res_chi_score.cpu() / (t



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
                    flex_rec_perturb = torch.zeros((n_rec, 3))
                    flex_lig_perturb = torch.zeros((n_lig, 3))

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

            if DEBUG:
                print("Before clamping, rec max perturbation:", flex_rec_perturb.max().item())
                print("Before clamping, lig max perturbation:", flex_lig_perturb.max().item())

            flex_rec_perturb = torch.clamp(flex_rec_perturb, min=-20, max=20)       # safe perturb
            flex_lig_perturb = torch.clamp(flex_lig_perturb, min=-20, max=20)       # safe perturb


            # to data list
            if type(complex_graphs) is not list:
                complex_graphs = complex_graphs.to("cpu").to_data_list()

            # print("flex_rec_perturb", flex_rec_perturb.mean())
            # print("flex_lig_perturb", flex_lig_perturb.mean())

            # apply transformations
            rec_i = 0
            lig_i = 0
            for i, complex_graph in enumerate(complex_graphs):
                new_data_list.extend([transform.apply_updates(complex_graph, tr_update=tr_perturb[i:i + 1], rot_update=rot_perturb[i:i + 1].squeeze(0),
                                    flex_rec_update=flex_rec_perturb[rec_i:rec_i+complex_graph['receptor'].pos.shape[0]], 
                                    flex_lig_update=flex_lig_perturb[lig_i:lig_i+complex_graph['ligand'].pos.shape[0]], 
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


        # keep down the last 4 steps
        if t_idx == args.num_steps - 1 or t_idx == args.num_steps - 2 or t_idx == args.num_steps - 3 or t_idx == args.num_steps - 4:
            for i in range(len(data_list)):
                write_pdb_each(f"{the_output_dir}/{names[i].split('/')[1]}_rs{args.seed}_step{args.num_steps - 1 - t_idx}.pdb", data_list[i]["receptor"].bb_atoms, data_list[i].rec_res_name_b, data_list[i]["ligand"].bb_atoms, data_list[i].lig_res_name_b, rwmode='a')
            
        
        if t_idx == args.num_steps - 1:
            if the_output_dir.split("_")[-1] != "test":
                for i in range(len(data_list)):
                    write_pdb_each(f"{the_output_dir}/{names[i].split('/')[1]}_rs{args.seed}.pdb", data_list[i]["receptor"].bb_atoms, data_list[i].rec_res_name_b, data_list[i]["ligand"].bb_atoms, data_list[i].lig_res_name_b, rwmode='w')
            
            updates_meter = evaluate_all_rmsds(new_data_list)
            # _,_,_,_ = updates_meter.summarize(verbose=True) #[????]
            complex_rmsd_list, interface_rmsd_list, names, true_lddt_list, fnat_list, ligand_rmsd_list, docking_quality_list = updates_meter.complex_rmsd_list, updates_meter.interface_rmsd_list, \
            updates_meter.names, updates_meter.true_lddt_list, updates_meter.fnat_list, updates_meter.ligand_rmsd_list, updates_meter.docking_quality_list

            # update all_results for IRMSD and CRMSD and tLDDT and LRMSD and fnat and docking_quality
            for i, name in enumerate(names):
                all_results[name]['IRMSD'].append(interface_rmsd_list[i])
                all_results[name]['CRMSD'].append(complex_rmsd_list[i])
                all_results[name]['LRMSD'].append(ligand_rmsd_list[i])
                all_results[name]['tLDDT'].append(true_lddt_list[i])
                all_results[name]['fnat'].append(fnat_list[i])
                all_results[name]['docking_quality'].append(docking_quality_list[i])

    



        # if t_idx == args.num_steps - 1:
        #     print("all_lddt_scores")

        #     for name in all_lddt_scores:
        #         print(name, all_lddt_scores[name])

        # updates_meter = evaluate_all_rmsds(new_data_list)
        # # _,_,_,_ = updates_meter.summarize(verbose=True) #[????]
        # complex_rmsd_list, interface_rmsd_list, names = updates_meter.complex_rmsd_list, updates_meter.interface_rmsd_list, updates_meter.names


        printt(f"Completed {t_idx} out of {args.num_steps} steps")



    #
    # lddt prediction [cur]
    # for t_fm in [0.0375, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125]:
    for t_fm in [0.0125]: #[0.6, 0.4, 0.2, 0.1]:
        print(f"t_fm: {t_fm}")
        t_idx = int(args.num_steps*t_fm) #########



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

        all_lddt_scores = {}
        gt_pred_bb_atoms = {}

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
            # convert to sigma space and save time
            if args.adaptive_schedule:
                flex_rec_s, flex_lig_s, rot_s, tr_s = transform.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr, adaptive_c=complex_graphs.adaptive_c)
            else:
                flex_rec_s, flex_lig_s, rot_s, tr_s = transform.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr)
            

            # flex_rec_s, flex_lig_s, rot_s, tr_s = transform.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr)
            

            device_for_set_time = complex_graphs["ligand"]["pos"].device if torch.cuda.is_available() else None
            
            if args.adaptive_schedule:
                if type(complex_graphs) is list:
                    for g, every_flex_rec_s, every_flex_lig_s in zip(complex_graphs, flex_rec_s, flex_lig_s):
                        set_time_flexFM_Rt(g, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device_for_set_time, t_fm_deform_rec=every_flex_rec_s, t_fm_deform_lig=every_flex_lig_s)
                else:
                    set_time_flexFM_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr, batch_size, device_for_set_time, t_fm_deform_rec=flex_rec_s, t_fm_deform_lig=flex_lig_s)
            else:
                if type(complex_graphs) is list:
                    for g in complex_graphs:
                        set_time_flexFM_Rt(g, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device_for_set_time)
                else:
                    set_time_flexFM_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr, batch_size, device_for_set_time)
                

            
            # if type(complex_graphs) is list:
            #     for g in complex_graphs:
            #         set_time_flexFM_Rt(g, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device_for_set_time)
            # else:
            #     set_time_flexFM_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr, batch_size, device_for_set_time)
            

            with torch.no_grad():
                outputs = model(complex_graphs)

            coord_r_score = outputs["coord_r_pred"]
            coord_l_score = outputs["coord_l_pred"]
            lddt_score = outputs["lddt_pred"]
            rot_score = outputs["rot_pred"]
            tr_score = outputs["tr_pred"]

            for name in complex_graphs.name:
                all_lddt_scores[name] = lddt_score[complex_graphs.name.index(name)].cpu().numpy()
            

            if type(complex_graphs) is not list:
                complex_graphs = complex_graphs.to("cpu").to_data_list()

            # bb_atoms
            for i, complex_graph in enumerate(complex_graphs):
                gt_pred_bb_atoms[complex_graph.name] = ((complex_graph["receptor"].bb_atoms_b, complex_graph["ligand"].bb_atoms_b), (complex_graph["receptor"].bb_atoms, complex_graph["ligand"].bb_atoms))

            
            
        # print("all_lddt_scores")

        # for name in all_lddt_scores:
        #     print(name, all_lddt_scores[name])
        
        if t_fm == 0.0125:
            # update all_results for pLDDT
            for name in all_lddt_scores:
                all_results[name]['pLDDT'].append(all_lddt_scores[name])
                all_results[name]['gt_pred_recLig_bb_atoms'].append(gt_pred_bb_atoms[name])
            # save all_results to the_output_dir/all_results.pkl
            with open(os.path.join(the_output_dir, f'all_results.pkl{pkl_ind}'), 'wb') as f:
                pickle.dump(all_results, f)
            
            # print("all_results")
            # for name in all_results:
            #     print(name, all_results[name])
            
            








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
    if all_results is not None:
        return data_list, all_results
    return data_list#, batch_size_to_return


def write_pdb_each(file_name, bb_coords_chain_R, sequence_chain_R, bb_coords_chain_L, sequence_chain_L, rwmode='w'):
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
def randomize_position(data_list, args):
    """
        Modify COPY of data_list objects
    """
    # data_list = copy.deepcopy(data_list)


    data_list = data_list.deepcopy()


    if not args.no_nm:
        # randomize nm coeffs
        for i, complex_graph in enumerate(data_list):

            nm_r_updates = torch.normal(mean=0, std=args.nm_s_max, size=(args.n_NM, 1))
            nm_l_updates = torch.normal(mean=0, std=args.nm_s_max, size=(args.n_NM, 1))
            complex_graph["receptor"].cur_nm = complex_graph["receptor"].u_nm
            complex_graph["ligand"].cur_nm = complex_graph["ligand"].u_nm

            flex_new_pos_l = complex_graph["ligand"].pos + ((complex_graph["ligand"].cur_nm).reshape((-1, args.n_NM))@nm_l_updates).reshape((-1, 3))
            flex_new_pos_r = complex_graph["receptor"].pos + ((complex_graph["receptor"].cur_nm).reshape((-1, args.n_NM))@nm_r_updates).reshape((-1, 3))
            
            complex_graph["ligand"].pos = flex_new_pos_l
            complex_graph["receptor"].pos = flex_new_pos_r

            data_list.set_graph(i, complex_graph)
    
    # if flex_only:
    if args.flex_only:
        for i, complex_graph in enumerate(data_list):
            data_list.set_graph(i, complex_graph)
        
        return data_list

    



    for i, complex_graph in enumerate(data_list):
        # randomize rotation
        # print(complex_graph)
        # print(complex_graph["ligand"])
        # print(complex_graph["ligand"].pos)
        # if type(complex_graph) == tuple: # TODO: remove
        #     complex_graph=complex_graph[0] 

        pos = complex_graph["ligand"].pos
        center = torch.mean(pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix())
        pos = (pos - center) @ random_rotation.T.float()

        if not args.no_nm:
            complex_graph["ligand"].cur_nm = torch.permute(torch.permute(complex_graph["ligand"].cur_nm, (0, 2, 1)) @ random_rotation.T.float(), (0, 2, 1))


        # random translation
        tr_update = torch.normal(0, args.tr_s_max, size=(1, 3))
        pos = pos + tr_update
        complex_graph["ligand"].pos = pos
        data_list.set_graph(i, complex_graph)

    return data_list




# [nm]
def randomize_position_AND_use_unbound(data_list, args):
    """
        Modify COPY of data_list objects
    """
    # data_list = copy.deepcopy(data_list)


    data_list = data_list.deepcopy()

    if args.adaptive_schedule:
        pkl_path='/scratch/user/rujieyin/DyDock/cat_updates.pkl'
        with open(pkl_path, 'rb') as f:
            pred_iRMSD_di = pickle.load(f)
        

    
    for i, complex_graph in enumerate(data_list):

        flex_new_pos_l = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask]
        flex_new_pos_r = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask]
        
        
        complex_graph["ligand"].pos = flex_new_pos_l
        complex_graph["receptor"].pos = flex_new_pos_r

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

        if args.res_rot_flex:
            # print(len(complex_graph["receptor"].mask), sum(complex_graph["receptor"].mask),'!!!')
            complex_graph['receptor'].bb_atoms = complex_graph['receptor'].bb_atoms_u[complex_graph["receptor"].mask]
            complex_graph['ligand'].bb_atoms = (complex_graph['ligand'].bb_atoms_u[complex_graph["ligand"].mask] - center[:,None,:]) @ ((random_rotation.T)[None,:,:]).float() + tr_update[:,None,:]

        if 'H_lig' in complex_graph:
            del complex_graph.H_lig
            del complex_graph.H_rec
        if 'Lambda' in complex_graph["receptor"]:
            del complex_graph["receptor"].Lambda
            del complex_graph["receptor"].P
            del complex_graph["ligand"].Lambda
            del complex_graph["ligand"].P
        if 'ts_rec' in complex_graph:
            del complex_graph.ts_rec
            del complex_graph.ts_lig

        if args.use_nm_msf_feat:
            # calculate msf
            complex_graph["receptor"].cur_nm = complex_graph["receptor"].b_nm # Shape: (n_res, 3, n_NM)
            complex_graph["receptor"].eig_vals = complex_graph["receptor"].eig_vals_b.float().reshape(1, -1) # Shape: (1, n_NM)
            msf = compute_msf(complex_graph["receptor"].cur_nm, complex_graph["receptor"].eig_vals.reshape(-1,))

            # q2 = torch.quantile(msf, 0.02)
            # q98 = torch.quantile(msf, 0.98)
            # msf_clipped = torch.clamp(msf, q2, q98)
            # data["receptor"].msf = msf_clipped # min_val + (msf_clipped - q5) / (q95 - q5) * (max_val - min_val)

            msf_log = torch.log(msf + 1e-6)  # add a small constant to avoid log(0)
            msf_normalized = (msf_log - msf_log.mean()) / msf_log.std()
            complex_graph["receptor"].msf = msf_normalized

            
            
            complex_graph["ligand"].cur_nm = complex_graph["ligand"].b_nm # Shape: (n_res, 3, n_NM)
            complex_graph["ligand"].eig_vals = complex_graph["ligand"].eig_vals_b.float().reshape(1, -1) # Shape: (1, n_NM)
            complex_graph["ligand"].msf = compute_msf(complex_graph["ligand"].cur_nm, complex_graph["ligand"].eig_vals.reshape(-1,))

            # q2 = torch.quantile(data["ligand"].msf, 0.02)
            # q98 = torch.quantile(data["ligand"].msf, 0.98)
            # msf_clipped = torch.clamp(data["ligand"].msf, q2, q98)
            # data["ligand"].msf = msf_clipped # min_val + (msf_clipped - q5) / (q95 - q5) * (max_val - min_val)

            msf_log = torch.log(complex_graph["ligand"].msf + 1e-6)  # add a small constant to avoid log(0)
            msf_normalized = (msf_log - msf_log.mean()) / msf_log.std()
            complex_graph["ligand"].msf = msf_normalized

            # print max and min msf
            # print(f"receptor max msf: {data['receptor'].msf.max()}; min msf: {data['receptor'].msf.min()}")
            # print(f"ligand max msf: {data['ligand'].msf.max()}; min msf: {data['ligand'].msf.min()}")
        

        if args.adaptive_schedule:
            name = complex_graph.name

            if name not in pred_iRMSD_di:
                print("name not in pred_iRMSD_di, so it is either in train or val set")
                if "irmsd" in complex_graph:
                    irmsd = complex_graph.irmsd.item()
                else:
                    irmsd = compute_irmsd(complex_graph['ligand'].pos_b_ori, complex_graph["receptor"].pos_b_ori, complex_graph['ligand'].pos_u[complex_graph['ligand'].mask], complex_graph['receptor'].pos_u[complex_graph['receptor'].mask], complex_graph.name)
            else:
                irmsd = pred_iRMSD_di[name].item()

            alpha = 0.99
            adaptive_c = ((-np.log(1-alpha)*np.log(3000.0))/(np.log(np.clip(irmsd, a_min=0.35, a_max=8.0))+np.log(100.0))) # c=10.4 to c=5.5
            complex_graph.adaptive_c = adaptive_c.item()


            



        data_list.set_graph(i, complex_graph)

    return data_list

def compute_msf(normal_mode_matrix, eigenvalues):
    ''' Compute the mean squared fluctuations (MSF) of a protein structure using normal modes.
    Args:
        normal_mode_matrix (torch.Tensor): The normal mode matrix of shape (n_res, 3, n_NM).
        eigenvalues (torch.Tensor): The eigenvalues of the normal modes of shape (n_NM,).
    Returns:
        torch.Tensor: The MSF of each residue in the protein structure of shape (n_res,).
    '''
    # Ensure eigenvalues are reshaped for broadcasting
    inv_eigenvalues = 1.0 / eigenvalues  # Shape: (n_NM,)

    # Compute the squared displacements: sum over x, y, z for each residue and mode
    squared_displacements = torch.einsum("rij,rij->rj", normal_mode_matrix, normal_mode_matrix)
    # Resulting shape: (n_res, n_NM)

    # Weight by the inverse eigenvalues and sum over modes (n_NM)
    msf = torch.einsum("rj,j->r", squared_displacements, inv_eigenvalues)
    # Resulting shape: (n_res,)
    return msf

def compute_irmsd(ligand_coors_true_ori, receptor_coors_true_ori, ligand_coors_pred, receptor_coors_pred, name):
    ''' Compute the interface root-mean-square deviation (iRMSD) between two protein structures.
    Args:
        ligand_coors_true_ori (torch.Tensor): The original coordinates of the ligand structure of shape (n_res_lig, 3).
        receptor_coors_true_ori (torch.Tensor): The original coordinates of the receptor structure of shape (n_res_rec, 3).
        ligand_coors_pred (torch.Tensor): The predicted coordinates of the ligand structure of shape (n_res_lig, 3).
        receptor_coors_pred (torch.Tensor): The predicted coordinates of the receptor structure of shape (n_res_rec, 3).
    Returns:
        torch.Tensor: The iRMSD between the two protein structures.
    '''
    # Compute the pairwise distances between ligand and receptor residues
    ligand_receptor_distance = torch.cdist(ligand_coors_true_ori, receptor_coors_true_ori)
    positive_tuple = torch.where(ligand_receptor_distance < 8.)

    if len(positive_tuple[0]) == 0:
        positive_tuple = torch.where(ligand_receptor_distance < 15.)
        print(f"There is no interface residue in the complex {name} with cutoff 8. Please check the input data.")

        

    # if len(positive_tuple[0]) == 0:
    #     print(f"There is no interface residue in the complex. Please check the input data.")
    

    # Remove duplicates while preserving order
    active_ligand, unique_ligand_indices = torch.unique(positive_tuple[0], return_inverse=True)
    active_receptor, unique_receptor_indices = torch.unique(positive_tuple[1], return_inverse=True)

    
    ligand_coors_pred = ligand_coors_pred[active_ligand, :]
    ligand_coors_true = ligand_coors_true_ori[active_ligand, :]

    receptor_coors_pred = receptor_coors_pred[active_receptor, :]
    receptor_coors_true = receptor_coors_true_ori[active_receptor, :]
    

    complex_coors_pred = torch.cat((ligand_coors_pred, receptor_coors_pred), dim=0)
    complex_coors_true = torch.cat((ligand_coors_true, receptor_coors_true), dim=0)

    R,t = kabsch_torch(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

    interface_rmsd = torch.sqrt(torch.mean((complex_coors_pred_aligned - complex_coors_true) ** 2))
    return interface_rmsd


    


