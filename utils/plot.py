import os

# file manipulation
import shutil
import numpy as np

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import torch.nn as nn
import torch.nn.functional as F

# gif
import imageio

def list_iter_numbers(dicts, key='n'):
    iteration_number_list = [dict[key] for dict in dicts]
    return iteration_number_list

def plot_samples(coords, file_name="./result_coords.png", is_coord_lim=False):
    # transfrom torch to numpy
    coords = coords.numpy()

    text_x = -1.5
    text_y = 8
    font_size_text = 20
    fs=17
    p = torch.distributions.Normal(0,3)
    
    plt.scatter(coords[:,1], coords[:,0], s=5, alpha=0.2,rasterized=True, color='C1')
    plt.legend(loc=0, fontsize=fs)
    plt.grid()
    
    if is_coord_lim:
        xlim = [-15,15]
        ylim = [-8,6]
        plt.xlim(xlim)
        plt.ylim(ylim)
        
    plt.tick_params(axis='both', labelsize=fs)
    plt.xlabel(r'$x_l', fontsize=font_size_text)
    
    '''
    vxx = torch.linspace(xlim[0],xlim[1],300)
    v_pdf = torch.exp(p.log_prob(vxx))
    '''

    plt.savefig(file_name, dpi=300)
    plt.clf()

def plot_bnn_regression(data, file_name="./result_regression.png", burn=10):
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    pred_y = data['pred_y']

    # plt
    plt.figure(figsize=(10,5))

    # all sampled function
    plt.plot(x_val, pred_y[burn:].squeeze().T, 'C0', alpha=0.051)

    # mean function
    plt.plot(x_val, np.mean(pred_y, axis=0).squeeze().T, 'C1', alpha=0.9)

    # upper std function
    plt.plot(x_val, np.mean(pred_y, axis=0).squeeze().T + np.std(pred_y, axis=0).squeeze().T, 'C1', alpha=0.8, linewidth=3)

    # lower std function
    plt.plot(x_val, np.mean(pred_y, axis=0).squeeze().T - np.std(pred_y, axis=0).squeeze().T, 'C1', alpha=0.8, linewidth=3)

    # training dataset
    plt.plot(x_train, y_train, '.C3', markersize=30, label='x_train', alpha=0.6)

    # legend
    plt.legend(fontsize=20)

    # lim
    plt.ylim([-5, 5])

    # save fig
    plt.savefig(file_name)


def plot_bnn_regression_uncertainty(data, tau_out, file_name="./result_regression_uncertainty.png", burn=10):
    """
        Input:
            Data : dict, all values are np.array
            file_name : string
            burn : the number of neglected samples
    """
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    pred_y = data['pred_y']

    # general variables
    mean = pred_y.mean(axis=0)
    std = pred_y.std(axis=0)

    # aleatoric
    std_al = (pred_y.var(axis=0) + (tau_out**(-1)))**0.5

    # Get upper and lower confidence bound
    lower, upper = (mean - 2*std).flatten(), (mean + 2*std).flatten()

    # Aleatoric
    lower_al, upper_al = (mean - 2*std_al).flatten(), (mean + 2*std_al).flatten()

    # plotting
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x_train, y_train, 'k*', rasterized=True)
    ax.plot(x_val, mean, 'b', rasterized=True)

    # shading
    ax.fill_between(x_val.flatten(), lower, upper, alpha=0.5, rasterized=True)
    ax.fill_between(x_val.flatten(), lower_al, upper_al, alpha=0.2, rasterized=True)

    # decorating
    # ax.set_ylim([-2, 2])
    # ax.set_xlim([-2, 2])
    plt.grid()
    ax.legend(['Observed Data', 'Mean', 'Epistemic', 'Aleatoric'], fontsize = 16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.tight_layout()

    # save fig
    plt.savefig(file_name)

def plot_gm_classification(data, file_name):
    x_train = data['x_train']
    y_train = data['y_train']
    
    pred_y = np.mean(data['pred_y'], axis=0)

    # scatter
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)

    # contour
    x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    plt.contourf(x1_test, x2_test, pred_y.reshape(100, 100), np.linspace(0, 1, 5), alpha=0.2)
    plt.colorbar()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable='box')
    plt.savefig(file_name)

def plot_banana_dist(loss_func, samples, file_name):

    # contour
    x1_test, x2_test = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
    # make two dimensional vector
    x_test = torch.FloatTensor(np.array([x1_test, x2_test])).view(2, -1).T

    # containers
    ll_test_list = []

    # predict
    for x_test_i in x_test:
        ll_test = loss_func(x_test_i).item()
        ll_test_list.append(ll_test)

    
    # reshape
    ll_test_list = np.reshape(np.array(ll_test_list), (500, 500))

    # contour plot
    plt.contourf(x1_test, x2_test, ll_test_list, np.linspace(-0.07, 0, 100), alpha=0.2)

    # colorbar
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")

    # scatter
    plt.scatter(samples[:,1], samples[:,0], s=5, alpha=0.8,rasterized=True, color='red')

    # tight_layout
    plt.tight_layout()

    # save fig
    plt.savefig(file_name)
    


def plot_bnn_learning_curves(loss_func, pred_list_tr, y_train, tau_out, file_name):    
    """
        Plotting negative log-likelihood
    """
    
    # data preprocecssing
    pred_list_tr = np.array(pred_list_tr)

    
    ll_full = np.zeros(pred_list_tr.shape[0])
    ll_full[0] = loss_func(pred_list_tr[0], y_train, tau_out).sum(0).item()
    for i in range(1, pred_list_tr.shape[0]):
        ll_full[i] = loss_func(pred_list_tr[:i].mean(axis=0), y_train, tau_out).sum(0).item()
        
    
    # plotting : learning curves
    f, ax1 = plt.subplots(1,1, figsize = (10,5))
    ax1.set_title('Training Negative Log-Likeilihood')
    ax1.plot(ll_full)
    ax1.grid()

    # savefig
    plt.savefig(file_name)

    
def plot_proposal_figure(coords,
    prop_coords, 
    hams, 
    proposed_hams,
    leapfrog_history,
    root_dir, 
    is_coupled=False, 
    is_coord_lim=True):

    # set subdirectory
    dir_name = os.path.join(root_dir, "proposal")
    os.makedirs(dir_name, exist_ok=True)
    
    # transfrom torch to numpy
    coords = coords.numpy()
    prop_coords = prop_coords.numpy()
    hams = hams.numpy()
    proposed_hams = proposed_hams.numpy()
    
    # file name
    file_name_list = []
    
    # For coupled
    if is_coupled:

        # For animations
        for i in range(0, coords.shape[0]-2, 2):
            # while counter
            count = 0
            
            while i in list_iter_numbers(leapfrog_history):
                try:
                    # leapfrog sample
                    leapfrog_dict = leapfrog_history.pop(0)
                    leapfrog_params_1 = leapfrog_dict['params_1'].numpy()
                    leapfrog_params_2 = leapfrog_dict['params_2'].numpy()
                    num_leapfrog = leapfrog_dict['num_leapfrog']
                    num_forwardsteps = leapfrog_dict['num_forward_steps']
                    leapfrog_error = leapfrog_dict['error_type']

                    num_leap_frog_1 = leapfrog_params_1.shape[0]
                    num_leap_frog_2 = leapfrog_params_2.shape[0]
                except:
                    continue
                
                # Background
                plt.scatter(background_points[:,1], background_points[:,0], s=5, alpha=0.2,rasterized=True, color='C1')
                
                # params 1
                for j in range(num_leap_frog_1):
                    # error leap frog
                    plt.scatter(leapfrog_params_1[j,1], leapfrog_params_1[j,0], s=5, alpha=0.3,rasterized=True, color='m')
                    # annotate
                    plt.annotate(j+1, (leapfrog_params_1[j,1], leapfrog_params_1[j,0]), alpha=0.3, size=5) # annotation
                
                # params 2                   
                for j in range(num_leap_frog_2):
                    # error leap frog
                    plt.scatter(leapfrog_params_2[j,1], leapfrog_params_2[j,0], s=5, alpha=0.3,rasterized=True, color='c')
                    # annotate
                    plt.annotate(j+1, (leapfrog_params_2[j,1], leapfrog_params_2[j,0]), alpha=0.3, size=5) # annotation
                    
                # file name
                file_name = "proposal_"+str(i)+ "_" + str(count) + ".png"
                file_name_list.append(file_name)
                
                if is_coord_lim:
                    xlim = [-15,15]
                    ylim = [-8,6]
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    
                # "is acceptance" shows x-axis
                is_accept_text = "{}_error_{}".format(i, leapfrog_error)
                plt.xlabel(is_accept_text, fontsize=10)
                
                # save fig
                plt.savefig(os.path.join(dir_name, file_name), dpi=300)
                
                # clr fig
                plt.clf()
                
                # counter 
                count += 1
            
            # braek
            if i+1 > coords.shape[0]:
                break
            
            # Background points
            background_points = coords[0:i+2]
        
            # Current point
            current_points_1 = coords[i:i+1]
            current_points_2 = coords[i+1:i+2]
            current_ham_1 = hams[i]
            current_ham_2 = hams[i+1]

            # Candidate
            candidate_1 = prop_coords[i:i+1]
            candidate_2 = prop_coords[i+1:i+2]
            candidate_ham_1 = proposed_hams[i]
            candidate_ham_2 = proposed_hams[i+1]

            # Background
            plt.scatter(background_points[:,1], background_points[:,0], s=5, alpha=0.2, rasterized=True, color='C1')

            # Current
            plt.scatter(current_points_1[:,1], current_points_1[:,0], s=7, alpha=0.7, rasterized=True, color='red')
            plt.scatter(current_points_2[:,1], current_points_2[:,0], s=7, marker="*", alpha=0.7, rasterized=True, color='red')

            # Candidate
            plt.scatter(candidate_1[:,1], candidate_1[:,0], s=7, alpha=0.7, rasterized=True, color='blue')
            plt.scatter(candidate_2[:,1], candidate_2[:,0], s=7, alpha=0.7, marker="*", rasterized=True, color='blue')

            # DEBUG
            temp1 = np.asscalar(current_points_1[:,1])
            temp2 = np.asscalar(current_points_1[:,0])
            temp3 = np.asscalar(candidate_1[:,1] - current_points_1[:,1])
            temp4 = np.asscalar(candidate_1[:,0] - current_points_1[:,0])
            
            temp5 = np.asscalar(current_points_2[:,1])
            temp6 = np.asscalar(current_points_2[:,0])
            temp7 = np.asscalar(candidate_2[:,1] - current_points_2[:,1])
            temp8 = np.asscalar(candidate_2[:,0] - current_points_2[:,0])

            # arrows
            plt.arrow(x=temp1, y=temp2, dx=temp3, dy=temp4, length_includes_head=True, zorder=0, head_length=0.3, head_width=0.15,
                      alpha=0.5, overhang = .2, width = .0003)
            
            plt.arrow(x=temp5, y=temp6, dx=temp7, dy=temp8, length_includes_head=True, zorder=0, head_length=0.3, head_width=0.15,
                      alpha=0.5, overhang = .2, width = .0003)

            if is_coord_lim:
                xlim = [-15,15]
                ylim = [-8,6]
                plt.xlim(xlim)
                plt.ylim(ylim)

            # "is acceptance" shows x-axis
            is_accept_text = "{}_Accept".format(i) if not np.array_equal(current_points_1, candidate_1) else "{}_Reject".format(i)
            is_accept_text = is_accept_text + "_Accept" if not np.array_equal(current_points_2, candidate_2) \
                else is_accept_text + "_Reject"
            plt.xlabel(is_accept_text, fontsize=10)
            
            # annotate
            plt.annotate(current_ham_1, (current_points_1[:,1], current_points_1[:,0]), alpha=0.5) # annotation
            plt.annotate(candidate_ham_1, (candidate_1[:,1], candidate_1[:,0]), alpha=0.5) # annotation
            plt.annotate(current_ham_2, (current_points_2[:,1], current_points_2[:,0]), alpha=0.5) # annotation
            plt.annotate(candidate_ham_2, (candidate_2[:,1], candidate_2[:,0]), alpha=0.5) # annotation
            
            # file name
            file_name = "proposal_"+str(i)+".png"
            file_name_list.append(file_name)

            # save fig
            plt.savefig(os.path.join(dir_name, file_name), dpi=300)

            # clr fig
            plt.clf()
    else:
        # For animations
        for i in range(coords.shape[0]-1):
            # while counter
            count = 0
            # error 
            while i in list_iter_numbers(leapfrog_history):
                try:
                    # leapfrog sample
                    leapfrog_dict = leapfrog_history.pop(0)
                    leapfrog_params = leapfrog_dict['params'].numpy() 
                    leapfrog_error = leapfrog_dict['error_type']
                    num_leap_frog = leapfrog_params.shape[0]
                except:
                    continue
                
                # Background
                plt.scatter(background_points[:,1], background_points[:,0], s=5, alpha=0.2,rasterized=True, color='C1')
                
                for j in range(num_leap_frog):
                    # error leap frog
                    plt.scatter(leapfrog_params[j,1], leapfrog_params[j,0], s=5, alpha=0.5,rasterized=True, color='green')
                    # annotate
                    plt.annotate(j+1, (leapfrog_params[j,1], leapfrog_params[j,0]), size=5) # annotation
                                        
                # file name
                file_name = "proposal_"+str(i-2)+ "_" + str(count) + ".png"
                file_name_list.append(file_name)
                
                if is_coord_lim:
                    xlim = [-15,15]
                    ylim = [-8,6]
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    
                # "is acceptance" shows x-axis
                is_accept_text = "{}_error_{}".format(i-2, leapfrog_error)
                plt.xlabel(is_accept_text, fontsize=10)

                # save fig
                plt.savefig(os.path.join(dir_name, file_name), dpi=300)
                
                # clr fig
                plt.clf()
                
                # counter 
                count += 1
                
            # Background points
            background_points = coords[0:i+1]
        
            # Current point
            current_points = coords[i:i+1]
            current_ham = hams[i]

            # Candidate
            candidate = prop_coords[i:i+1]
            candidate_ham = proposed_hams[i]

            # Background
            plt.scatter(background_points[:,1], background_points[:,0], s=5, alpha=0.2,rasterized=True, color='C1')

            # Current
            plt.scatter(current_points[:,1], current_points[:,0], s=7, alpha=0.7, rasterized=True, color='red')

            # Candidate
            plt.scatter(candidate[:,1], candidate[:,0], s=7, alpha=0.7, rasterized=True, color='blue')

            # DEBUG
            temp1 = np.asscalar(current_points[:,1])
            temp2 = np.asscalar(current_points[:,0])
            temp3 = np.asscalar(candidate[:,1] - current_points[:,1])
            temp4 = np.asscalar(candidate[:,0] - current_points[:,0])

            # arrows
            plt.arrow(x=temp1,y=temp2, dx=temp3, dy=temp4, length_includes_head=True, zorder=0, head_length=0.3, head_width=0.15,
                      alpha=0.5, overhang = .2, width = .0003)

            if is_coord_lim:
                xlim = [-15,15]
                ylim = [-8,6]
                plt.xlim(xlim)
                plt.ylim(ylim)

            # "is acceptance" shows x-axis
            is_accept_text = "{}_Accept".format(i) if not np.array_equal(current_points, candidate) else "Reject"
            plt.xlabel(is_accept_text, fontsize=10)
            
            # annotate
            plt.annotate(current_ham, (current_points[:,1], current_points[:,0])) # annotation
            plt.annotate(candidate_ham, (candidate[:,1], candidate[:,0])) # annotation
            
            # file name
            file_name = "proposal_"+str(i)+".png"
            file_name_list.append(file_name)

            # save fig
            plt.savefig(os.path.join(dir_name, file_name), dpi=300)
            
            # clr fig
            plt.clf()
                
    # making a gif
    images = []
    
    for file_name in file_name_list:
        images.append(imageio.imread(os.path.join(dir_name, file_name)))
        
    # delete all file
    shutil.rmtree(dir_name)
    
    # duration
    kargs = {'duration' : 1}
        
    # make gif
    imageio.mimsave(os.path.join(root_dir, "animation.gif"), images, "GIF", **kargs)
    
        
    
        
    
        
        
        

    
    
    
    


