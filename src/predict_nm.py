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
from geom_utils import set_time_nm, NoiseTransform_nm

from evaluation.compute_rmsd_nm import evaluate_all_rmsds, evaluate_all_ori_rmsds


EPS = 1e-6


DEBUG_inv = False

DEBUG_steps_80 = False # True # 

DEBUG_no_random = True # False #


DEBUG_save = True # False #


def predict_coord(data_list, model, args, epoch=0, visualize_first_n_samples=0,
           visualization_dir="./visualization", in_batch_size=None, output_coeffs=False):
    """
        Run reverse process
    """
    DEBUG = False

    if DEBUG_steps_80:
        args.num_steps = 80
        args.actual_steps = 80
    
    # [norm]
    if args.arcsinh_norm:
        args.nm_s_max = args.nm_s_max_arcsinh_norm

        args.tr_s_max = args.tr_s_max_arcsinh_norm

    if in_batch_size is None:
        in_batch_size = args.batch_size
        
    # switch to eval mode
    model.eval()

    # stores various noise-related utils
    transform = NoiseTransform_nm(args)

    # diffusion timesteps
    # timesteps = get_timesteps(args.num_steps)

    # # Prepare for visualizations
    # visualize_first_n_samples = min(visualize_first_n_samples, len(data_list))
    # graph_gts = [data_list[i] for i in range(visualize_first_n_samples)]
    # visualization_values = [data_list.get_visualization_values(index=i) for i in range(visualize_first_n_samples)]
    # four_letter_pdb_names = [get_four_letters_pdb_identifier(graph_gt.name) for graph_gt in graph_gts]
    # visualization_dirs = create_visualization_directories(visualization_dir, epoch, four_letter_pdb_names)

    # # sde, rand_ub
    # args.starting_from_random_flexed_unbound = True #False #  !!!!
    # args.ode = False #True #  !!!!

    # ode, ub
    args.starting_from_random_flexed_unbound = False #True #  !!!!
    args.ode = True #False #  !!!!


    # randomize original position and COPY data_list
    data_list = use_unbound_coord(data_list, args)

    # save ori_data_list
    ori_data_list = copy.deepcopy(data_list)

    # if type(ori_data_list) is not list:
    #     ori_data_list = ori_data_list.to("cpu").to_data_list()


    ori_meter = evaluate_all_ori_rmsds(data_list)
    _,_,_,_ = ori_meter.summarize(verbose=True)

    complex_rmsd_list, interface_rmsd_list, ori_names, recovery_rmsd_prec_list = ori_meter.complex_rmsd_list, ori_meter.interface_rmsd_list, ori_meter.names, ori_meter.recovery_rmsd_prec_list
    ori_ind_rmsd_list = ori_meter.ind_rmsd_list

    if DEBUG_save:
        with open("ori_rmsds.pkl", 'wb') as f:
            pickle.dump([complex_rmsd_list, interface_rmsd_list, ori_names, recovery_rmsd_prec_list], f)

    names = ori_meter.names


    array = np.array(ori_meter.complex_rmsd_list)

    # Get the indices of the smallest 10 elements
    sml_indices = np.argsort(array)[:5]
    lrg_indices = np.argsort(array)[-5:]

    print("sml_indices", sml_indices, "names", [names[i] for i in sml_indices])
    print("lrg_indices", lrg_indices, "names", [names[i] for i in lrg_indices])

    # construct a dict 
    dict = {"sml_indices": {}, "lrg_indices": {}}
    for i in sml_indices:
        dict["sml_indices"][i] = {'name': names[i], 'complex_rmsd': array[i]}
    for i in lrg_indices:
        dict["lrg_indices"][i] = {'name': names[i], 'complex_rmsd': array[i]}
    
    # # save the dict
    # with open("deform_NM_100_sml_lrg_complexes.pkl", 'wb') as f:
    #     pickle.dump(dict, f)

    cat_updates = {}

    # predict
    args.num_steps = 1 # !!!

    for t_idx in range(args.num_steps):
        # create new loader with current step graphs
        if torch.cuda.is_available() and args.num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader
        test_loader = loader(data_list, batch_size=args.batch_size)
        
        new_data_list = []  # updated every step


        # # uses same schedule for all noise
        # cur_t = timesteps[t_idx]
        # if t_idx == args.num_steps - 1:
        #     dt = cur_t
        # else:
        #     dt = cur_t - timesteps[t_idx+1]
        
        # ######
        # print("dt", dt)


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

            if DEBUG:
                print("complex_graphs !!!!!!!!!!")
                print(complex_graphs)
            


            with torch.no_grad():
                outputs = model(complex_graphs)
            # tr_score = outputs["tr_pred"].cpu()
            # rot_score = outputs["rot_pred"].cpu()

            if 'coord' in args.model_type:
                nm_r_score = outputs["coord_r_pred"].cpu()
                nm_l_score = outputs["coord_l_pred"].cpu()


                # print(complex_graphs['receptor'].batch)

                # separate the batch data
                rec_batch = complex_graphs['receptor'].batch
                lig_batch = complex_graphs['ligand'].batch

                # Find common unique graph indices between receptor and ligand graphs
                common_graphs = rec_batch.unique()[torch.isin(rec_batch.unique(), lig_batch.unique())]
                
                delta_r = []
                delta_l = []
                for graph_idx in common_graphs:

            
                    delta_r.append(nm_r_score[rec_batch == graph_idx])
                    delta_l.append(nm_l_score[lig_batch == graph_idx])

                # print(delta_r)

                nm_r_score = delta_r
                nm_l_score = delta_l





                # print("coord_r_pred.shape", nm_r_score.shape)
                # print("coord_l_pred.shape", nm_l_score.shape)


            else:
                nm_r_score = outputs["nm_r_pred"].cpu()
                nm_l_score = outputs["nm_l_pred"].cpu()


            nm_r_update = nm_r_score
            nm_l_update = nm_l_score

            
            if DEBUG:
                
                # print("tr_score.shape", tr_score.shape)
                # print("rot_score.shape", rot_score.shape)
                print("nm_r_score.shape", nm_r_score.shape)
                print("nm_l_score.shape", nm_l_score.shape)


            # to data list
            if type(complex_graphs) is not list:
                complex_graphs = complex_graphs.to("cpu").to_data_list()

            # #
            # ######### updates #########
            
            # for ii, com_g in enumerate(complex_graphs):
                    
            #     # print(nm_r_update.shape, nm_l_update.shape)
                    
            #     # print(com_g['ligand'].b_to_u_nmCoeffs.shape)
            #     # print(com_g.name)

            #     if com_g.name not in cat_updates:
            #         cat_updates[com_g.name] = {}
            #         cat_updates[com_g.name]['b_to_u_nmCoeffs'] = torch.cat((com_g['receptor'].b_to_u_nmCoeffs.reshape(1, -1), com_g['ligand'].b_to_u_nmCoeffs.reshape(1, -1)), dim=0)
            #         # print(torch.cat((com_g['receptor'].b_to_u_nmCoeffs.reshape(1, -1), com_g['ligand'].b_to_u_nmCoeffs.reshape(1, -1)), dim=0).shape)
            #         cat_updates[com_g.name]['updates'] = []
                    
            #     cat_updates[com_g.name]['updates'].append(torch.cat([nm_r_update[ii].reshape(1, -1), nm_l_update[ii].reshape(1, -1)], dim=0))

            #     # print(torch.cat([nm_r_update[ii].reshape(1, -1), nm_l_update[ii].reshape(1, -1)], dim=0).shape)



            # apply transformations
            

            # 
            # if flex_only
            if args.flex_only:
                
                for i, data in enumerate(complex_graphs):

                    # print(data["ligand"].pos.shape)

                    new_graph = transform.apply_updates(data,
                            coord_r_updates=nm_r_update[i], 
                            coord_l_updates=nm_l_update[i],
                            )

                    new_data_list.append(new_graph)
                
        # update starting point for next step
        assert len(new_data_list) == len(data_list)
        data_list = new_data_list

        updates_meter = evaluate_all_rmsds(new_data_list)
        _,_,_,_ = updates_meter.summarize(verbose=True)
        complex_rmsd_list, interface_rmsd_list, names, recovery_rmsd_prec_list = updates_meter.complex_rmsd_list, updates_meter.interface_rmsd_list, updates_meter.names, updates_meter.recovery_rmsd_prec_list
        ind_rmsd_list = updates_meter.ind_rmsd_list


        # print([[i, j] for i, j in zip(ori_names, names)])

        all_means = []
        for ori_pair, pred_pair in zip(ori_ind_rmsd_list, ind_rmsd_list):
            # print(ori_pair, pred_pair)

            print("mean", np.mean(np.array(ori_pair) - np.array(pred_pair)))
            print("diff", np.array(ori_pair) - np.array(pred_pair))

            all_means.append(np.mean(np.array(ori_pair) - np.array(pred_pair)))
        
        print(np.mean(all_means))





        # if DEBUG_save:
        #     with open(f"predict_rmsds.pkl", 'wb') as f:
        #         pickle.dump([complex_rmsd_list, interface_rmsd_list, names, recovery_rmsd_prec_list], f)

        # for i, nn in enumerate(names):

        #     if 'crmsd' not in cat_updates[nn]:
        #         cat_updates[nn]['crmsd'] = []
        #         cat_updates[nn]['irmsd'] = []
            
        #     cat_updates[nn]['crmsd'].append(complex_rmsd_list[i])
        #     cat_updates[nn]['irmsd'].append(interface_rmsd_list[i])


        printt(f"Completed {t_idx} out of {args.num_steps} steps")

        # if DEBUG_inv and t_idx == args.num_steps - 1:
        #     print("ligand pos after sampling:", data_list[0]["ligand"]["pos"][:5])
        #     print("receptor pos after sampling:", data_list[0]["receptor"]["pos"][:5])

        # Cut last diffusion steps short because they tend to oeverfit
        if t_idx >= args.actual_steps - 1:
            break
        # === end of timestep ===

    if output_coeffs:
        return data_list, cat_updates

    return data_list#, batch_size_to_return




# [nm]
def predict(data_list, model, args, epoch=0, visualize_first_n_samples=0,
           visualization_dir="./visualization", in_batch_size=None, output_coeffs=False):
    """
        Run reverse process
    """
    DEBUG = False

    if DEBUG_steps_80:
        args.num_steps = 80
        args.actual_steps = 80
    
    # [norm]
    if args.arcsinh_norm:
        args.nm_s_max = args.nm_s_max_arcsinh_norm

        args.tr_s_max = args.tr_s_max_arcsinh_norm

    if in_batch_size is None:
        in_batch_size = args.batch_size
        
    # switch to eval mode
    model.eval()

    # # stores various noise-related utils
    # transform = NoiseTransform_nm(args)

    # diffusion timesteps
    # timesteps = get_timesteps(args.num_steps)

    # # Prepare for visualizations
    # visualize_first_n_samples = min(visualize_first_n_samples, len(data_list))
    # graph_gts = [data_list[i] for i in range(visualize_first_n_samples)]
    # visualization_values = [data_list.get_visualization_values(index=i) for i in range(visualize_first_n_samples)]
    # four_letter_pdb_names = [get_four_letters_pdb_identifier(graph_gt.name) for graph_gt in graph_gts]
    # visualization_dirs = create_visualization_directories(visualization_dir, epoch, four_letter_pdb_names)

    # # sde, rand_ub
    # args.starting_from_random_flexed_unbound = True #False #  !!!!
    # args.ode = False #True #  !!!!

    # ode, ub
    args.starting_from_random_flexed_unbound = False #True #  !!!!
    args.ode = True #False #  !!!!


    # randomize original position and COPY data_list
    data_list = randomize_position_AND_use_unbound(data_list, args)

    # save ori_data_list
    ori_data_list = copy.deepcopy(data_list)

    # if type(ori_data_list) is not list:
    #     ori_data_list = ori_data_list.to("cpu").to_data_list()


    # ori_meter = evaluate_all_ori_rmsds(data_list)
    # _,_,_,_ = ori_meter.summarize(verbose=True)

    # complex_rmsd_list, interface_rmsd_list, names, recovery_rmsd_prec_list = ori_meter.complex_rmsd_list, ori_meter.interface_rmsd_list, ori_meter.names, ori_meter.recovery_rmsd_prec_list

    # if DEBUG_save:
    #     with open("ori_rmsds.pkl", 'wb') as f:
    #         pickle.dump([complex_rmsd_list, interface_rmsd_list, names, recovery_rmsd_prec_list], f)

    # names = ori_meter.names


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

        




    # if DEBUG:
    #     print("data_list after randomize_position_AND_use_unbound !!!!!!!!!!")
    #     print(data_list)

    # # For visualization
    # for i in range(visualize_first_n_samples):
    #     write_pdb(visualization_values[i], graph_gts[i], "receptor",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-receptor.pdb")
    #     write_pdb(visualization_values[i], graph_gts[i], "ligand",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-gt.pdb")
    #     write_pdb(visualization_values[i], data_list[i], "ligand",
    #           f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-0.pdb")

    # # determine batch_size
    # batch_size=args.batch_size
    # while batch_size > 2:
    #     try:
    #         test_loader = DataLoader(data_list, batch_size=batch_size)
    #         for complex_graphs in test_loader:
    #             complex_graphs = complex_graphs.cuda(args.gpu)
    #             set_time(complex_graphs, 0, 0, 0, batch_size, complex_graphs["ligand"]["pos"].device)
    #             with torch.no_grad():
    #                 outputs = model(complex_graphs)
    #                 #outputs = model(complex_graphs)
    #             print('Ran model')
    #             break
    #         break
    #     except RuntimeError as e:
    #         if 'out of memory' in str(e):
    #             print('| WARNING: ran out of memory, Reducing batch size')
    #             for p in model.parameters():
    #                 if p.grad is not None:
    #                     del p.grad  # free some memory
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             batch_size = batch_size // 2
    #             print('Reduced batch size')

    # for p in model.parameters():
    #     if p.grad is not None:
    #         del p.grad  # free some memory
    # torch.cuda.empty_cache()
    # gc.collect()
    # # Reducing it one more time to be safe
    # #if batch_size > 2:
    # #    batch_size = batch_size // 2
    # print(f'Using batch size: {batch_size}')
    # batch_size_to_return = batch_size
        
    cat_updates = {}

    # predict
    args.num_steps = 1 # !!!

    for t_idx in range(args.num_steps):
        # create new loader with current step graphs
        if torch.cuda.is_available() and args.num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader
        test_loader = loader(data_list, batch_size=args.batch_size)
        
        new_data_list = []  # updated every step


        # # uses same schedule for all noise
        # cur_t = timesteps[t_idx]
        # if t_idx == args.num_steps - 1:
        #     dt = cur_t
        # else:
        #     dt = cur_t - timesteps[t_idx+1]
        
        # ######
        # print("dt", dt)


        # print("!!!!!!!! rank: ", args.rank, "test_loader.data: ", test_loader.data, "!!!!!!!!")

        for com_idx, complex_graphs in enumerate(test_loader):

            # if DEBUG_no_random:
            #     args.no_random = True

            # if DEBUG_inv and t_idx == 0:
            #     args.no_random = True # no drifting term


            #     # random_rotation_inv = torch.from_numpy(R.random().as_matrix())
            #     random_rotation_inv = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

            #     print("ligand pos before random rotation:", complex_graphs["ligand"]["pos"][:5])
            #     print("receptor pos before random rotation:", complex_graphs["receptor"]["pos"][:5])
            #     print("random_rotation_inv:", random_rotation_inv)

            #     for part in ["ligand", "receptor"]:


            #         pos = complex_graphs[part]["pos"]
            #         pos = pos @ random_rotation_inv.T.float()
            #         complex_graphs[part]["pos"] = pos

            #         complex_graphs[part]["cur_nm"] = torch.permute(torch.permute(complex_graphs[part]["cur_nm"], (0, 2, 1)) @ random_rotation_inv.T.float(), (0, 2, 1))

            #     print("ligand pos after random rotation:", complex_graphs["ligand"]["pos"][:5])
            #     print("receptor pos after random rotation:", complex_graphs["receptor"]["pos"][:5])

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

            # # convert to sigma space and save time
            # tr_s, rot_s, nm_s = transform.noise_schedule(
            #     cur_t, cur_t, cur_t)
            
            # # ASB
            # g_t = transform.noise_schedule.g(cur_t)


            # device_for_set_time = complex_graphs["ligand"]["pos"].device if torch.cuda.is_available() and args.num_gpu == 1 else None
            # if type(complex_graphs) is list:
            #     for g in complex_graphs:
            #         set_time_nm(g, cur_t, cur_t, cur_t, 1, device_for_set_time)
            # else:
            #     set_time_nm(complex_graphs, cur_t, cur_t, cur_t, batch_size, device_for_set_time)
            
            if DEBUG:
                print("complex_graphs !!!!!!!!!!")
                print(complex_graphs)
            


            with torch.no_grad():
                outputs = model(complex_graphs)
            names = complex_graphs.name
            irmsd_preds = outputs["irmsd_pred"].cpu()

            for name, irmsd_pred in zip(names, irmsd_preds):
                print(name, "irmsd_pred", irmsd_pred)
                cat_updates[name] = irmsd_pred

            # tr_score = outputs["tr_pred"].cpu()
            # rot_score = outputs["rot_pred"].cpu()

            # if 'coord' in args.model_type:
            #     nm_r_score = outputs["coord_r_pred"].cpu()
            #     nm_l_score = outputs["coord_l_pred"].cpu()
            # else:
            #     nm_r_score = outputs["nm_r_pred"].cpu()
            #     nm_l_score = outputs["nm_l_pred"].cpu()


            # nm_r_update = nm_r_score.cpu()
            # nm_l_update = nm_l_score.cpu()

            # if args.ASB:
            #     # tr_pred, rot_pred, nm_r_pred, nm_l_pred, doobs_h_r_pred, doobs_h_l_pred
            #     doobs_h_r_score = outputs["doobs_h_r_pred"].cpu()
            #     doobs_h_l_score = outputs["doobs_h_l_pred"].cpu()


            # if DEBUG:
                
            #     # print("tr_score.shape", tr_score.shape)
            #     # print("rot_score.shape", rot_score.shape)
            #     print("nm_r_score.shape", nm_r_score.shape)
            #     print("nm_l_score.shape", nm_l_score.shape)
            
            # if DEBUG_inv:
            #     # print("tr_score:", tr_score)
            #     # print("rot_score:", rot_score)
            #     print("nm_r_score:", nm_r_score)
            #     print("nm_l_score:", nm_l_score)

            # # translation gradient (?)
            # tr_scale = torch.sqrt(
            #     2 * torch.log(torch.tensor(args.tr_s_max /
            #                                args.tr_s_min)))
            # tr_g = tr_s * tr_scale

            # # rotation gradient (?)
            # rot_scale = torch.sqrt(
            #         2 * torch.log(torch.tensor(args.rot_s_max /
            #                                args.rot_s_min)))
            # rot_g = rot_s * rot_scale

            # # nm gradient (?) for r/l
            # if not args.no_nm:
            #     nm_scale = torch.sqrt(
            #         2 * torch.log(torch.tensor(args.nm_s_max /
            #                                 args.nm_s_min)))
            #     nm_g = nm_s * nm_scale

            # # actual update for tr, rot, nm (all)
            # if args.ode:
            #     tr_update = (0.5 * tr_g**2 * dt * tr_score)
            #     rot_update = (0.5 * rot_score * dt * rot_g**2)
                
                
            #     if not args.no_nm:
            #         if args.interpolation_nm:
            #             # if tr_sigma < 6 and protein_dynamic:
            #             nm_r_update = nm_r_score.cpu() / (args.num_steps-t_idx+args.num_steps*0.25)
            #             nm_l_update = nm_l_score.cpu() / (args.num_steps-t_idx+args.num_steps*0.25)
            #             # else:
            #             # res_tr_perturb = torch.zeros((n, 3))
            #             # res_rot_perturb = torch.zeros((n, 3))
            #             # res_chi_perturb = torch.zeros((n, 5))

            #         else:
            #             nm_r_update = (0.5 * nm_g**2 * dt * nm_r_score)
            #             nm_l_update = (0.5 * nm_g**2 * dt * nm_l_score)
            # else:
            #     if args.no_final_noise and t_idx == args.num_steps-1:
            #         tr_z = torch.zeros((batch_size, 3))
            #         rot_z = torch.zeros((batch_size, 3))
            #         if not args.no_nm:
            #             nm_r_z = torch.zeros((batch_size, args.n_NM))
            #             nm_l_z = torch.zeros((batch_size, args.n_NM))
            #     elif args.no_random:
            #         tr_z = torch.zeros((batch_size, 3))
            #         rot_z = torch.zeros((batch_size, 3))
            #         if not args.no_nm:
            #             nm_r_z = torch.zeros((batch_size, args.n_NM))
            #             nm_l_z = torch.zeros((batch_size, args.n_NM))
            #     else:
            #         tr_z = torch.normal(0, 1, size=(batch_size, 3))
            #         rot_z = torch.normal(0, 1, size=(batch_size, 3))
            #         if not args.no_nm:
            #             nm_r_z = torch.normal(0, 1, size=(batch_size, args.n_NM))
            #             nm_l_z = torch.normal(0, 1, size=(batch_size, args.n_NM))

            #     tr_update = (tr_g**2 * dt * tr_score)
            #     tr_update = tr_update + (tr_g * np.sqrt(dt) * tr_z)

            #     rot_update = (rot_score * dt * rot_g**2)
            #     rot_update = rot_update + (rot_g * np.sqrt(dt) * rot_z)

            #     if not args.no_nm:
            #         if args.interpolation_nm:
            #             # if tr_sigma < 6 and protein_dynamic:
            #             nm_r_update = nm_r_score.cpu() / (args.num_steps-t_idx+args.num_steps*0.25)
            #             nm_l_update = nm_l_score.cpu() / (args.num_steps-t_idx+args.num_steps*0.25)
            #             # res_chi_perturb = res_chi_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
            #             # # res_tr_perturb = res_tr_score.cpu() / (t_idx+inference_steps*0.1)
            #             # # res_rot_perturb = res_rot_score.cpu() / (t_idx+inference_steps*0.1)
            #             # # res_chi_perturb = res_chi_score.cpu() / (t_idx+inference_steps*0.1)
            #             # else:
            #             # res_tr_perturb = torch.zeros((n, 3))
            #             # res_rot_perturb = torch.zeros((n, 3))
            #             # res_chi_perturb = torch.zeros((n, 5))
            #         elif args.ASB:
            #             nm_r_update = (g_t**2 * dt * nm_r_score)
            #             nm_r_update = nm_r_update + (g_t * np.sqrt(dt) * nm_r_z)

            #             nm_l_update = (g_t**2 * dt * nm_l_score)
            #             nm_l_update = nm_l_update + (g_t * np.sqrt(dt) * nm_l_z)

            #         else:
            #             nm_r_update = (nm_g**2 * dt * nm_r_score)
            #             nm_r_update = nm_r_update + (nm_g * np.sqrt(dt) * nm_r_z)

            #             nm_l_update = (nm_g**2 * dt * nm_l_score)
            #             nm_l_update = nm_l_update + (nm_g * np.sqrt(dt) * nm_l_z)

            # if args.temp_sampling != 1.0:
            #     tr_sigma_data = np.exp(args.temp_sigma_data_tr * np.log(args.tr_s_max) + (1 - args.temp_sigma_data_tr) * np.log(args.tr_s_min))
            #     lambda_tr = (tr_sigma_data + tr_s) / (tr_sigma_data + tr_s / args.temp_sampling)
            #     tr_update = (tr_g ** 2 * dt * (lambda_tr + args.temp_sampling * args.temp_psi / 2) * tr_score.cpu() + tr_g * np.sqrt(dt * (1 + args.temp_psi)) * tr_z).cpu()

            #     rot_sigma_data = np.exp(args.temp_sigma_data_rot * np.log(args.rot_s_max) + (1 - args.temp_sigma_data_rot) * np.log(args.rot_s_min))
            #     lambda_rot = (rot_sigma_data + rot_s) / (rot_sigma_data + rot_s / args.temp_sampling)
            #     rot_update = (rot_g ** 2 * dt * (lambda_rot + args.temp_sampling * args.temp_psi / 2) * rot_score.cpu() + rot_g * np.sqrt(dt * (1 + args.temp_psi)) * rot_z).cpu()

            #     if not args.no_nm:
            #         nm_sigma_data = np.exp(args.temp_sigma_data_nm * np.log(args.nm_s_max) + (1 - args.temp_sigma_data_nm) * np.log(args.nm_s_min))
            #         lambda_nm = (nm_sigma_data + nm_s) / (nm_sigma_data + nm_s / args.temp_sampling)
            #         nm_r_update = (nm_g ** 2 * dt * (lambda_nm + args.temp_sampling * args.temp_psi / 2) * nm_r_score.cpu() + nm_g * np.sqrt(dt * (1 + args.temp_psi)) * nm_r_z).cpu()
            #         nm_l_update = (nm_g ** 2 * dt * (lambda_nm + args.temp_sampling * args.temp_psi / 2) * nm_l_score.cpu() + nm_g * np.sqrt(dt * (1 + args.temp_psi)) * nm_l_z).cpu()


            # # to data list
            # if type(complex_graphs) is not list:
            #     complex_graphs = complex_graphs.to("cpu").to_data_list()

            # #
            # ######### updates #########
            
            # for ii, com_g in enumerate(complex_graphs):
                    
            #     # print(nm_r_update.shape, nm_l_update.shape)
                    
            #     # print(com_g['ligand'].b_to_u_nmCoeffs.shape)
            #     # print(com_g.name)

            #     if com_g.name not in cat_updates:
            #         cat_updates[com_g.name] = {}
            #         cat_updates[com_g.name]['b_to_u_nmCoeffs'] = torch.cat((com_g['receptor'].b_to_u_nmCoeffs.reshape(1, -1), com_g['ligand'].b_to_u_nmCoeffs.reshape(1, -1)), dim=0)
            #         # print(torch.cat((com_g['receptor'].b_to_u_nmCoeffs.reshape(1, -1), com_g['ligand'].b_to_u_nmCoeffs.reshape(1, -1)), dim=0).shape)
            #         cat_updates[com_g.name]['updates'] = []
                    
            #     cat_updates[com_g.name]['updates'].append(torch.cat([nm_r_update[ii].reshape(1, -1), nm_l_update[ii].reshape(1, -1)], dim=0))

                # print(torch.cat([nm_r_update[ii].reshape(1, -1), nm_l_update[ii].reshape(1, -1)], dim=0).shape)

                
 

                    


            # apply transformations
            

            # # 
            # # if flex_only
            # if args.flex_only:
            #     if 'coord' in args.model_type:
            #         for i, data in enumerate(complex_graphs):
            #             new_graph = transform.apply_updates(data,
            #                     coord_r_updates=nm_r_update[i:i+1], 
            #                     coord_l_updates=nm_l_update[i:i+1],
            #                     )

            #             new_data_list.append(new_graph)
            #     else:
            #         for i, data in enumerate(complex_graphs):
            #             new_graph = transform.apply_updates(data,
            #                     nm_r_updates=nm_r_update[i:i+1].reshape((-1, 1)), 
            #                     nm_l_updates=nm_l_update[i:i+1].reshape((-1, 1)),
            #                     )

            #             new_data_list.append(new_graph)

                

                
                






                

                    

            # elif args.tr_only:
            #     for i, data in enumerate(complex_graphs):
            #         new_graph = transform.apply_updates(data,
            #                 tr_update=tr_update[i:i+1],
            #                 )

            #         new_data_list.append(new_graph)
            # elif args.no_nm:
            #     for i, data in enumerate(complex_graphs):
            #         new_graph = transform.apply_updates(data,
            #                 tr_update = tr_update[i:i+1],
            #                 rot_update = rot_update[i:i+1].squeeze(0),
            #                 )

            #         new_data_list.append(new_graph)
            # else:
            #     for i, data in enumerate(complex_graphs):
            #         new_graph = transform.apply_updates(data,
            #                 tr_update[i:i+1],
            #                 rot_update[i:i+1].squeeze(0),
            #                 nm_r_update[i:i+1].reshape((-1, 1)), 
            #                 nm_l_update[i:i+1].reshape((-1, 1)),
            #                 )

            #         new_data_list.append(new_graph)
                    


            # === end of batch ===
            #printt(f'finished batch {com_idx}')

        # for i in range(visualize_first_n_samples):
        #     write_pdb(visualization_values[i], new_data_list[i], "ligand",
        #               f"{visualization_dirs[i]}/{four_letter_pdb_names[i]}-ligand-{t_idx + 1}.pdb")

    #     # update starting point for next step
    #     assert len(new_data_list) == len(data_list)
    #     data_list = new_data_list

    #     updates_meter = evaluate_all_rmsds(new_data_list)
    #     _,_,_,_ = updates_meter.summarize(verbose=True)
    #     complex_rmsd_list, interface_rmsd_list, names, recovery_rmsd_prec_list = updates_meter.complex_rmsd_list, updates_meter.interface_rmsd_list, updates_meter.names, updates_meter.recovery_rmsd_prec_list

    #     if DEBUG_save:
    #         with open(f"predict_rmsds.pkl", 'wb') as f:
    #             pickle.dump([complex_rmsd_list, interface_rmsd_list, names, recovery_rmsd_prec_list], f)

    #     for i, nn in enumerate(names):

    #         if 'crmsd' not in cat_updates[nn]:
    #             cat_updates[nn]['crmsd'] = []
    #             cat_updates[nn]['irmsd'] = []
            
    #         cat_updates[nn]['crmsd'].append(complex_rmsd_list[i])
    #         cat_updates[nn]['irmsd'].append(interface_rmsd_list[i])


    #     printt(f"Completed {t_idx} out of {args.num_steps} steps")

    #     if DEBUG_inv and t_idx == args.num_steps - 1:
    #         print("ligand pos after sampling:", data_list[0]["ligand"]["pos"][:5])
    #         print("receptor pos after sampling:", data_list[0]["receptor"]["pos"][:5])

    #     # Cut last diffusion steps short because they tend to oeverfit
    #     if t_idx >= args.actual_steps - 1:
    #         break
    #     # === end of timestep ===

    # if output_coeffs:
    #     return data_list, cat_updates

    # if "cat_updates.pkl" exists: update the dict instead of overwriting it
    if os.path.exists("cat_updates.pkl"):
        cat_updates_old = pickle.load(open("cat_updates.pkl", "rb"))
        for key in cat_updates.keys():
            cat_updates_old[key] = cat_updates[key]
            
        cat_updates = cat_updates_old

    pickle.dump(cat_updates, open("cat_updates.pkl", "wb"))

    return data_list#, batch_size_to_return


# def create_visualization_directories(top_visualization_dir, epoch, pdb_names):
#     visualization_dirs = [f"{top_visualization_dir}/epoch-{epoch}/{pdb_name}" for pdb_name in pdb_names]
#     for directory in visualization_dirs:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     return visualization_dirs

# def get_four_letters_pdb_identifier(pdb_name):
#     return pdb_name.split('/')[-1].split('.')[0]


# def write_pdb(item, graph, part, path):
#     lines = to_pdb_lines(item, graph, part)
#     with open(path, "w") as file:
#         file.writelines(lines)


# def to_pdb_lines(visualization_values, graph, part):
#     assert part in ("ligand", "receptor", "both"), "Part should be ligand or receptor"
#     parts = ["ligand", "receptor"] if part == "both" else [part]

#     lines = []
#     for part in parts:
#         this_vis_values = visualization_values[part]
#         this_vis_values = {k: v.strip() if type(v) is str else v for k, v in this_vis_values.items()}
#         for i, resname in enumerate(this_vis_values["resname"]):
#             xyz = graph[part].pos[i]

#             line = f'ATOM  {i + 1:>5} {this_vis_values["atom_name"][i]:>4} '
#             line = line + f'{resname} {this_vis_values["chain"][i]}{this_vis_values["residue"][i]:>4}    '.replace("<Chain id=", "").replace(">", "")
#             line = line + f'{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}'
#             line = line + '  1.00  0.00          '
#             line = line + f'{this_vis_values["element"][i]:>2} 0\n'
#             lines.append(line)

#     return lines

# [nm]
# def get_timesteps(inference_steps):
#     return np.linspace(1, 0, inference_steps + 1)[:-1]


# # [nm]
# def randomize_position(data_list, args):
#     """
#         Modify COPY of data_list objects
#     """
#     # data_list = copy.deepcopy(data_list)


#     data_list = data_list.deepcopy()


#     if not args.no_nm:
#         # randomize nm coeffs
#         for i, complex_graph in enumerate(data_list):

#             nm_r_updates = torch.normal(mean=0, std=args.nm_s_max, size=(args.n_NM, 1))
#             nm_l_updates = torch.normal(mean=0, std=args.nm_s_max, size=(args.n_NM, 1))
#             complex_graph["receptor"].cur_nm = complex_graph["receptor"].u_nm
#             complex_graph["ligand"].cur_nm = complex_graph["ligand"].u_nm

#             flex_new_pos_l = complex_graph["ligand"].pos + ((complex_graph["ligand"].cur_nm).reshape((-1, args.n_NM))@nm_l_updates).reshape((-1, 3))
#             flex_new_pos_r = complex_graph["receptor"].pos + ((complex_graph["receptor"].cur_nm).reshape((-1, args.n_NM))@nm_r_updates).reshape((-1, 3))
            
#             complex_graph["ligand"].pos = flex_new_pos_l
#             complex_graph["receptor"].pos = flex_new_pos_r

#             data_list.set_graph(i, complex_graph)
    
#     # if flex_only:
#     if args.flex_only:
#         for i, complex_graph in enumerate(data_list):
#             data_list.set_graph(i, complex_graph)
        
#         return data_list

    



#     for i, complex_graph in enumerate(data_list):
#         # randomize rotation
#         # print(complex_graph)
#         # print(complex_graph["ligand"])
#         # print(complex_graph["ligand"].pos)
#         # if type(complex_graph) == tuple: # TODO: remove
#         #     complex_graph=complex_graph[0] 

#         pos = complex_graph["ligand"].pos
#         center = torch.mean(pos, dim=0, keepdim=True)
#         random_rotation = torch.from_numpy(R.random().as_matrix())
#         pos = (pos - center) @ random_rotation.T.float()

#         if not args.no_nm:
#             complex_graph["ligand"].cur_nm = torch.permute(torch.permute(complex_graph["ligand"].cur_nm, (0, 2, 1)) @ random_rotation.T.float(), (0, 2, 1))


#         # random translation
#         tr_update = torch.normal(0, args.tr_s_max, size=(1, 3))
#         pos = pos + tr_update
#         complex_graph["ligand"].pos = pos
#         data_list.set_graph(i, complex_graph)

#     return data_list


# [nm]
def use_unbound_coord(data_list, args):
    """
        Modify COPY of data_list objects
    """
    # data_list = copy.deepcopy(data_list)


    data_list = data_list.deepcopy()

    
    # randomize nm coeffs
    for i, complex_graph in enumerate(data_list):

        # use unbound nm (aligned part)
        complex_graph["receptor"].cur_nm = complex_graph["receptor"].b_nm
        complex_graph["ligand"].cur_nm = complex_graph["ligand"].b_nm

        # eigen vals
        complex_graph["receptor"].eig_vals = 1/(complex_graph["receptor"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["receptor"].pos_b.shape[0], -1) + EPS)
        complex_graph["ligand"].eig_vals = 1/(complex_graph["ligand"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["ligand"].pos_b.shape[0], -1) + EPS)


        # # OLD
        # complex_graph["receptor"].eig_vals = complex_graph["receptor"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["receptor"].pos_b.shape[0], -1)
        # complex_graph["ligand"].eig_vals = complex_graph["ligand"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["ligand"].pos_b.shape[0], -1)

    
        flex_new_pos_l = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask]
        flex_new_pos_r = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask]
    
        complex_graph["ligand"].pos = flex_new_pos_l
        complex_graph["receptor"].pos = flex_new_pos_r

        if 'x_ori' in complex_graph["ligand"]:
            # use unbound node features
            complex_graph["ligand"].x = complex_graph["ligand"].x_ori[complex_graph["ligand"].mask]
            complex_graph["receptor"].x = complex_graph["receptor"].x_ori[complex_graph["receptor"].mask]

            # use bound edge index
            complex_graph["ligand", "contact", "ligand"].edge_index = complex_graph["ligand"].edge_index_b
            complex_graph["receptor", "contact", "receptor"].edge_index = complex_graph["receptor"].edge_index_b
        else:
            pass
        
        data_list.set_graph(i, complex_graph)
    
    # if flex_only:
    if args.flex_only:
        # print("here 1 !!!!!!!!!!")
        return data_list

# [nm]
def randomize_position_AND_use_unbound(data_list, args):
    """
        Modify COPY of data_list objects
    """
    # data_list = copy.deepcopy(data_list)


    data_list = data_list.deepcopy()

    for i, complex_graph in enumerate(data_list):
        complex_graph["receptor"].num_nodes = complex_graph["receptor"].pos_b_ori.shape[0]
        complex_graph["ligand"].num_nodes = complex_graph["ligand"].pos_b_ori.shape[0]

        complex_graph["ligand"].pos = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask]
        complex_graph["receptor"].pos = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask]

        if 'x_ori' in complex_graph["ligand"]:
            # use unbound node features
            complex_graph["ligand"].x = complex_graph["ligand"].x_ori[complex_graph["ligand"].mask]
            complex_graph["receptor"].x = complex_graph["receptor"].x_ori[complex_graph["receptor"].mask]

        # eliminate all other attributes
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
        
        data_list.set_graph(i, complex_graph)
    
    return data_list
        

    # if (not args.no_nm) and args.interpolation_nm:
    #     # randomize nm coeffs
    #     for i, complex_graph in enumerate(data_list):


    #         complex_graph["receptor"].cur_nm = complex_graph["receptor"].b_nm
    #         complex_graph["ligand"].cur_nm = complex_graph["ligand"].b_nm

            
    #         flex_new_pos_l = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask]
    #         flex_new_pos_r = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask]
            
            
    #         complex_graph["ligand"].pos = flex_new_pos_l
    #         complex_graph["receptor"].pos = flex_new_pos_r

    #         if 'x_ori' in complex_graph["ligand"]:
    #             # use unbound node features
    #             complex_graph["ligand"].x = complex_graph["ligand"].x_ori[complex_graph["ligand"].mask]
    #             complex_graph["receptor"].x = complex_graph["receptor"].x_ori[complex_graph["receptor"].mask]

    #             # use unbound edge index
    #             complex_graph["ligand", "contact", "ligand"].edge_index = complex_graph["ligand"].edge_index_b
    #             complex_graph["receptor", "contact", "receptor"].edge_index = complex_graph["receptor"].edge_index_b
    #         else:
    #             pass
            

    #         # !!!!!!!!!!
    #         # NEW
    #         complex_graph["receptor"].eig_vals = 1/(complex_graph["receptor"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["receptor"].pos_b.shape[0], -1)  + EPS)
    #         complex_graph["ligand"].eig_vals = 1/(complex_graph["ligand"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["ligand"].pos_b.shape[0], -1) + EPS)

    #         # # OLD
    #         # complex_graph["receptor"].eig_vals = complex_graph["receptor"].eig_vals_u.float().reshape(1, -1).expand(complex_graph["receptor"].pos_u.shape[0], -1)
    #         # complex_graph["ligand"].eig_vals = complex_graph["ligand"].eig_vals_u.float().reshape(1, -1).expand(complex_graph["ligand"].pos_u.shape[0], -1)

            

    #         data_list.set_graph(i, complex_graph)


    # elif not args.no_nm:
    #     # randomize nm coeffs
    #     for i, complex_graph in enumerate(data_list):

    #         # print("here 2 !!!!!!!!!!", i, "here 2 !!!!!!!!!!")

    #         # random generation of nm updates according in the max noise level
    #         nm_r_updates = torch.normal(mean=0, std=args.nm_s_max, size=(args.n_NM, 1))
    #         nm_l_updates = torch.normal(mean=0, std=args.nm_s_max, size=(args.n_NM, 1))


    #         if args.arcsinh_norm:
                
                
    #             if not args.no_range_limit:
                
    #                 if args.arcsinh_train_resample:
                    
    #                     while True:
                            
    #                         out_of_range_mask = (nm_l_updates < np.arcsinh(args.nm_update_min)) | (nm_l_updates > np.arcsinh(args.nm_update_max))
                            
    #                         if not torch.any(out_of_range_mask):
    #                             break

    #                         num_out_of_range = out_of_range_mask.sum()

    #                         resampled_values = torch.normal(mean=0, std=args.nm_s_max, size=(num_out_of_range, 1))

    #                         nm_l_updates[out_of_range_mask] = resampled_values.reshape((-1))
                        
    #                     while True:
    #                         out_of_range_mask = (nm_r_updates < np.arcsinh(args.nm_update_min)) | (nm_r_updates > np.arcsinh(args.nm_update_max))
                            
    #                         if not torch.any(out_of_range_mask):
    #                             break

    #                         num_out_of_range = out_of_range_mask.sum()

    #                         resampled_values = torch.normal(mean=0, std=args.nm_s_max, size=(num_out_of_range, 1))
                            
    #                         nm_r_updates[out_of_range_mask] = resampled_values.reshape((-1))
                    
    #                 else:
                    
    #                     # clamping
    #                     nm_l_updates = torch.clamp(nm_l_updates, np.arcsinh(args.nm_update_min), np.arcsinh(args.nm_update_max))
    #                     nm_r_updates = torch.clamp(nm_r_updates, np.arcsinh(args.nm_update_min), np.arcsinh(args.nm_update_max))
                    
    #                 nm_l_updates = torch.sinh(nm_l_updates)
    #                 nm_r_updates = torch.sinh(nm_r_updates)
                    

    #         # use unbound nm (aligned part)
    #         complex_graph["receptor"].cur_nm = complex_graph["receptor"].b_nm
    #         complex_graph["ligand"].cur_nm = complex_graph["ligand"].b_nm

    #         # eigen vals
    #         complex_graph["receptor"].eig_vals = 1/(complex_graph["receptor"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["receptor"].pos_b.shape[0], -1) + EPS)
    #         complex_graph["ligand"].eig_vals = 1/(complex_graph["ligand"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["ligand"].pos_b.shape[0], -1) + EPS)


    #         # # OLD
    #         # complex_graph["receptor"].eig_vals = complex_graph["receptor"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["receptor"].pos_b.shape[0], -1)
    #         # complex_graph["ligand"].eig_vals = complex_graph["ligand"].eig_vals_b.float().reshape(1, -1).expand(complex_graph["ligand"].pos_b.shape[0], -1)

            



    #         if not args.starting_from_random_flexed_unbound:
    #             flex_new_pos_l = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask]
    #             flex_new_pos_r = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask]
            
    #         else:

    #             flex_new_pos_l = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask] + ((complex_graph["ligand"].cur_nm).reshape((-1, args.n_NM))@nm_l_updates).reshape((-1, 3))
    #             flex_new_pos_r = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask] + ((complex_graph["receptor"].cur_nm).reshape((-1, args.n_NM))@nm_r_updates).reshape((-1, 3))
                
            
    #         complex_graph["ligand"].pos = flex_new_pos_l
    #         complex_graph["receptor"].pos = flex_new_pos_r

    #         if 'x_ori' in complex_graph["ligand"]:
    #             # use unbound node features
    #             complex_graph["ligand"].x = complex_graph["ligand"].x_ori[complex_graph["ligand"].mask]
    #             complex_graph["receptor"].x = complex_graph["receptor"].x_ori[complex_graph["receptor"].mask]

    #             # use bound edge index
    #             complex_graph["ligand", "contact", "ligand"].edge_index = complex_graph["ligand"].edge_index_b
    #             complex_graph["receptor", "contact", "receptor"].edge_index = complex_graph["receptor"].edge_index_b
    #         else:
    #             pass
            

    #         data_list.set_graph(i, complex_graph)
        
    # # if flex_only:
    # if args.flex_only:
    #     # print("here 1 !!!!!!!!!!")
    #     return data_list
    
    # # translation only 
    # if args.tr_only:
    #     for i, complex_graph in enumerate(data_list):
    #         # randomize rotation
    #         # print(complex_graph)
    #         # print(complex_graph["ligand"])
    #         # print(complex_graph["ligand"].pos)
    #         # if type(complex_graph) == tuple: # TODO: remove
    #         #     complex_graph=complex_graph[0]

            
    #         complex_graph["ligand"].pos = complex_graph["ligand"].pos_b_ori
    #         complex_graph["receptor"].pos = complex_graph["receptor"].pos_b_ori

    #         if 'x_ori' in complex_graph["ligand"]:
    #             # use unbound node features
    #             complex_graph["ligand"].x = complex_graph["ligand"].x_ori[complex_graph["ligand"].mask]
    #             complex_graph["receptor"].x = complex_graph["receptor"].x_ori[complex_graph["receptor"].mask]

    #             # use unbound edge index
    #             complex_graph["ligand", "contact", "ligand"].edge_index = complex_graph["ligand"].edge_index_b
    #             complex_graph["receptor", "contact", "receptor"].edge_index = complex_graph["receptor"].edge_index_b
    #         else:
                
    #             pass

    #         pos = complex_graph["ligand"].pos
    #         center = torch.mean(pos, dim=0, keepdim=True)
    #         pos = (pos - center)

    #         # random translation
    #         tr_update = torch.normal(0, args.tr_s_max, size=(1, 3))
    #         pos = pos + tr_update
    #         complex_graph["ligand"].pos = pos

    #         if args.invariant_tr_output:
    #             num_rows = complex_graph['ligand'].pos.shape[0]

    #             a = torch.zeros(num_rows, 3)
    #             a[:, 0] = 1
    #             b = torch.zeros(num_rows, 3)
    #             b[:, 1] = 1
    #             c = torch.zeros(num_rows, 3)
    #             c[:, 2] = 1
    #             nm_like_coordAxis = torch.cat((a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1)), dim=-1).float().to(complex_graph["ligand"].pos.device) # (batch['ligand'].pos.shape[0], 3, 3)

    #             if args.enable_random_init_axis_in_inv_tr:

                    
    #                 # generate random rotation
    #                 rand_rot_mat = torch.from_numpy(R.random().as_matrix()).float()

    #                 nm_like_coordAxis = torch.permute(torch.permute(nm_like_coordAxis, (0, 2, 1)) @ rand_rot_mat.T, (0, 2, 1))

    #             complex_graph["ligand"].cur_tr_nm = nm_like_coordAxis

    #         data_list.set_graph(i, complex_graph)

    #     return data_list


    # for i, complex_graph in enumerate(data_list):
    #     # randomize rotation
    #     # print(complex_graph)
    #     # print(complex_graph["ligand"])
    #     # print(complex_graph["ligand"].pos)
    #     # if type(complex_graph) == tuple: # TODO: remove
    #     #     complex_graph=complex_graph[0]

    #     if args.no_nm:

    #         if args.train_label_pos_b_ori: # this affects sampling

    #             complex_graph["ligand"].pos = complex_graph["ligand"].pos_b_ori
    #             complex_graph["receptor"].pos = complex_graph["receptor"].pos_b_ori

                

    #             if 'x_ori' in complex_graph["ligand"]:
    #                 # use unbound node features
    #                 complex_graph["ligand"].x = complex_graph["ligand"].x_ori[complex_graph["ligand"].mask]
    #                 complex_graph["receptor"].x = complex_graph["receptor"].x_ori[complex_graph["receptor"].mask]

    #                 # use unbound edge index
    #                 complex_graph["ligand", "contact", "ligand"].edge_index = complex_graph["ligand"].edge_index_b
    #                 complex_graph["receptor", "contact", "receptor"].edge_index = complex_graph["receptor"].edge_index_b
    #             else:
                    
    #                 pass
    #         else:
                    
    #             complex_graph["ligand"].pos = complex_graph["ligand"].pos_u[complex_graph["ligand"].mask]
    #             complex_graph["receptor"].pos = complex_graph["receptor"].pos_u[complex_graph["receptor"].mask]

    #             if 'x_ori' in complex_graph["ligand"]:
    #                 # use unbound node features
    #                 complex_graph["ligand"].x = complex_graph["ligand"].x_ori[complex_graph["ligand"].mask]
    #                 complex_graph["receptor"].x = complex_graph["receptor"].x_ori[complex_graph["receptor"].mask]

    #                 # use unbound edge index
    #                 complex_graph["ligand", "contact", "ligand"].edge_index = complex_graph["ligand"].edge_index_b
    #                 complex_graph["receptor", "contact", "receptor"].edge_index = complex_graph["receptor"].edge_index_b
    #             else:
                    
    #                 pass
            
    #         if args.invariant_tr_output:

    #             num_rows = complex_graph['ligand'].pos.shape[0]

    #             a = torch.zeros(num_rows, 3)
    #             a[:, 0] = 1
    #             b = torch.zeros(num_rows, 3)
    #             b[:, 1] = 1
    #             c = torch.zeros(num_rows, 3)
    #             c[:, 2] = 1
    #             nm_like_coordAxis = torch.cat((a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1)), dim=-1).float().to(complex_graph["ligand"].pos.device) # (batch['ligand'].pos.shape[0], 3, 3)

    #             if args.enable_random_init_axis_in_inv_tr:

    #                 rand_rot_mat = torch.from_numpy(R.random().as_matrix()).float()

    #                 nm_like_coordAxis = torch.permute(torch.permute(nm_like_coordAxis, (0, 2, 1)) @ rand_rot_mat.T, (0, 2, 1))

    #             complex_graph["ligand"].cur_tr_nm = nm_like_coordAxis
            
    #         # else:


    #         #     complex_graph["ligand"].pos = complex_graph["ligand"].pos_u
    #         #     complex_graph["receptor"].pos = complex_graph["receptor"].pos_u

    #         #     if 'x_ori' in complex_graph["ligand"]:
    #         #         # use unbound node features
    #         #         complex_graph["ligand"].x = complex_graph["ligand"].x_ori
    #         #         complex_graph["receptor"].x = complex_graph["receptor"].x_ori

    #         #         # use unbound edge index
    #         #         complex_graph["ligand", "contact", "ligand"].edge_index = complex_graph["ligand"].edge_index_u
    #         #         complex_graph["receptor", "contact", "receptor"].edge_index = complex_graph["receptor"].edge_index_u
    #         #     else:
                    
    #         #         pass

    #     pos = complex_graph["ligand"].pos
    #     center = torch.mean(pos, dim=0, keepdim=True)
    #     random_rotation = torch.from_numpy(R.random().as_matrix())
    #     pos = (pos - center) @ random_rotation.T.float()

    #     if not args.no_nm:
    #         complex_graph["ligand"].cur_nm = torch.permute(torch.permute(complex_graph["ligand"].cur_nm, (0, 2, 1)) @ random_rotation.T.float(), (0, 2, 1))


    #     # random translation
    #     tr_update = torch.normal(0, args.tr_s_max, size=(1, 3))
    #     pos = pos + tr_update
    #     complex_graph["ligand"].pos = pos


    #     if args.invariant_tr_output: # not necessary
    #         complex_graph["ligand"].cur_tr_nm = torch.permute(torch.permute(complex_graph["ligand"].cur_tr_nm, (0, 2, 1)) @ random_rotation.T.float(), (0, 2, 1))

    #     data_list.set_graph(i, complex_graph)

    # return data_list

