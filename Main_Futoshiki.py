# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import pickle
from tqdm import tqdm
from Scripts import guess_device

import multiprocessing as mp
import concurrent.futures

from Futoshiki import *

grad_save = 0
W_save = 0
nb_save= 0
def train_PLL(args, game_utils, device):
    print("\n \n \n TRAINING")
    
    #### MODEL ####
            #create model
    model = Net(game_utils.nb_var, game_utils.nb_val, game_utils.nb_features,
                hidden_size=args.hidden_size, 
                nblocks=args.nblocks)

    # print model size
    param_size = 0
    for param in model.parameters():
            param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    if hasattr(game_utils, "net"):
        model.net = game_utils.net
    #load saved model if instructed to do so
    if args.saved_dict!="":
        print("loading model from disk", args.saved_dict)
        model.load_state_dict(torch.load(args.saved_dict,  map_location=device))

    #model = Net.VerySimpleNet(81,9,4, device)
    #instanciate optimizer and scheduler

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.weight_decay)

    #optimizer = torch.optim.SGD(model.parameters(), 
    #                             lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = args.scheduler_factor, patience = args.scheduler_patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = args.scheduler_factor, patience = args.scheduler_patience)


    print("\n Training with the following parameters: \n" 
            + str(args))

    print("\n \n")
    global grad_save, nb_save
    batch_size = args.batch_size
    #### TRAINING ####
    for epoch in range(1, args.epoch_max):
        print("EPOCH ", epoch, " on ", args.epoch_max)
            
    
        PLL_epoch,PLL1_epoch, loss_epoch, L1_epoch, unary_L1_epoch = 0, 0, 0, 0, 0
        model.train()
        data_iterator = game_utils.get_data() 
        grad_save = 0
        nb_save = 0
        for batch_idx in tqdm(range(args.train_size)):
        
            queries, target, infos = next(data_iterator)
            optimizer.zero_grad()
            NN_input = queries.to(device)  # bs, nb_var, nb_var, nb_feature
            y_true = target.type(torch.LongTensor).to(device) #bs,nb_var
            W, unary = model(NN_input, device, unary = args.unary)
            L1 = torch.linalg.vector_norm(W, ord = 1)  # L1 penalty on predicted cost
            #L1 = torch.sum(torch.abs(W)*(torch.abs(W)>3))
            unary_L1 = torch.linalg.vector_norm(unary, ord=1)
            
            #PLL = -EPLL_utils.PLL_all(W, y_true, nb_neigh = args.k, hints_logit=unary)
            PLL1 = -EPLL_utils.PLL_all2(W, y_true, nb_neigh = args.k, hints_logit=unary, mask_width=args.order, nb_rand_masks = args.nb_rand_masks, nb_rand_tuples = args.nb_rand_tuples)
            PLL=PLL1
            PLL_epoch += PLL.item()
            PLL1_epoch += PLL1.item()
            unary_L1_epoch += unary_L1.item()*args.reg_term_unary
            L1_epoch += L1.item()*args.reg_term
             
            def funcgrad(grad):
                global W_save, grad_save,nb_save
                print(grad)

            loss = PLL1 + args.reg_term * L1 + args.reg_term_unary * unary_L1
            #unary.register_hook(funcgrad)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        optimizer.zero_grad()

        test_loss = 0
        test_acc = 0
        # VALIDATION
        data_iterator = game_utils.get_data(validation=True) 
        for batch_idx in range(args.valid_size): 
            queries, target, infos = next(data_iterator)
            NN_input = queries.to(device)  # bs, nb_var, nb_var, nb_feature
            W, unary = model(NN_input, device, unary = args.unary)
            y_true = target.type(torch.LongTensor).to(device) #bs,nb_var
            acc, PLL = PLL_all2(W, y_true, hints_logit = unary, val=True, mask_width=args.order, nb_rand_masks = args.nb_rand_masks, nb_rand_tuples = args.nb_rand_tuples)
            PLL = -PLL
            loss = PLL #+ args.reg_term * L1
            test_loss += loss.item()
            test_acc += loss.item()
            if(batch_idx == 0 and args.debug>0):
                print("DEBUG")
                
                # print solution
                #print(y_true[0].reshape(9,9)+1)
                # print hints
                #print(infos[0,:].reshape(9,9))
                # print various cost matrix (for sudoku)
                print(target[0,:].reshape(9,9)+1)
                print(queries[0,0,3])
                print(W[0,3,4].cpu().detach().numpy().round(2))
                print(W[0,3,13].cpu().detach().numpy().round(2))
                print(W[0,3,23].cpu().detach().numpy().round(2))
                print(unary[0,3].cpu().detach().numpy().round(2))
                
                print(unary[0,0].cpu().detach().numpy().round(2))
                #print(np.trunc(unary[0,8].cpu().detach().numpy()))
                #print(np.trunc(unary[0,0].cpu().detach().numpy()))
                #print(np.trunc(unary[0,1].cpu().detach().numpy()))
                print("FINdebug")

                # dump the problem to disk, so that it can be solved using cmd
                Wb = W[0].cpu().detach().numpy()
                unaryd = unary[0].cpu().detach().numpy()
                unaryd = unaryd-unaryd.min(axis=1)[:,None]
                Wb = Wb-Wb.min(axis=-1).min(axis=-1)[:,:,None,None]
                Wb = Wb*(Wb>args.threshold)
                solver(Wb,unaryd,onlyDump=True)
                

        scheduler.step(loss_epoch)
        print(f'Training loss: {loss_epoch:.1f} ( =  PLL term : {PLL_epoch:.1f} + PLL1 : {PLL1_epoch:.1f} + L1 term : {L1_epoch:.1f} + unary_L1 : {unary_L1_epoch:.1f} || Testing loss = {test_loss:.1f} Testing accuracy = {test_acc}')
        print("current lr", optimizer.param_groups[0]['lr'], " ( min lr", args.min_lr," )")
        print("END OF EPOCH")

        if(optimizer.param_groups[0]['lr']<args.min_lr):
            print("lr is too small, stopping training")
            break

    print("\n END OF TRAINING \n")
    return model

def test(args, model, game_utils, device):
    print("\n \n \n TESTING \n")
    print("Test in progress (can take several minutes)")
    
    results = []
            
    print("Starting parallel processes for solving: ")
    # we parallelize the resolution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processes = []
        data_iterator = game_utils.get_data(test=True) 

        for batch_idx in tqdm(range(args.test_size)):
        
            queries, target, infos = next(data_iterator)
            NN_input = queries.to(device)  # bs, nb_var, nb_var, nb_feature
            y_true = target.type(torch.LongTensor).to(device)
            
            W, unary = model(NN_input, device, unary = args.unary)
            unary = unary-unary.min(axis=-1)[0].detach()[:,:,None]
            W =W -W.min(axis=-1)[0].min(axis=-1)[0].detach()[:,:,:,None,None]
            Wb = W[0].cpu().detach().numpy()
            unaryd = unary[0].cpu().detach().numpy()
            Wb = Wb*(Wb>args.threshold)
            proc = executor.submit(solve_parallel,game_utils.check_valid,queries[0].cpu().detach().numpy(),target[0].cpu().detach().numpy(), infos[0].cpu().detach().numpy() ,Wb, unaryd, debug = args.debug)
            processes.append(proc)
        print(" Collecting results ( be patient )")
        for process_idx in tqdm(range(args.test_size)):
            query, target, info, W, unary, res, game = processes[process_idx].result()
            results.append(res)

    result = np.array(results)
    print("Test done, games succesfully solved : ", np.sum(result),"on", args.test_size)
    #print(results)
    return results

def solve_parallel (check_valid_ft, query, target, info, W, unary, debug=2):
    ret, game = check_valid_ft(query, target, info, W, unary, debug)
    return query, target, info, W, unary, ret, game

def main():            

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, prog='EPLL', description='learn a game rule representation using only valid exemples.')
    
    argparser.add_argument("--path_to_data", type=str, default="Futoshiki/databases/", help="path for loading training data") 
    argparser.add_argument("--game_type", type=str, default="Sudoku_hints", help="type of game ( Futoshiki, Sudoku, Sudoku_hints, Sudoku_grounding, Sudoku_visual )") 
    #with 1oM: Data_raw/one_of_many/
    argparser.add_argument("--hidden_size", type=int, default=128, help="width of hidden layers") 
    argparser.add_argument("--nblocks", type=int, default=5, help="number of blocks of 2 layers in ResNet") 
    argparser.add_argument("--lr", type=float, default=0.001, help="learning rate") 
    argparser.add_argument("--weight_decay", type=float, default=0, help="weight_decay") 
    argparser.add_argument("--reg_term", type=float, default=0, help="L1 regularization on costs")
    argparser.add_argument("--reg_term_unary", type=float, default=0, help="L1 regularization on unary")
    argparser.add_argument("--unary", type=int, default=1, help="Use unary costs")
    argparser.add_argument("--k", type=int, default=40, help="E-PLL parameter") 
    argparser.add_argument("--order", type=int, default=2, help="order of the PLL parameter ( mask width )")  # mask width
    argparser.add_argument("--nb_rand_masks", type=int, default=100, help="stochastic PLL parameter : number of random masks")  
    argparser.add_argument("--nb_rand_tuples", type=int, default=40, help="stochastic PLL parameter : number of random tuple of values to consider")  
    argparser.add_argument("--batch_size", type=int, default=8, help="training batch size") 
    argparser.add_argument("--epoch_max", type=int, default=200, help="maximum number of epochs") 
    argparser.add_argument("--train_size", type=int, default=1000, help="number of training samples") 
    argparser.add_argument("--valid_size", type=int, default=64, help="number of validation samples") 
    argparser.add_argument("--test_size", type=int, default=70, help="number of test samples") 
    argparser.add_argument("--saved_dict", type=str, default="", help="load weights saved from a previous execution") 
    argparser.add_argument("--save_path", type=str, default="./model_save.pk", help="location to which the weights are saved when training is over") 
    argparser.add_argument("--seed", type=int, default=np.random.randint(0, 2000), help="manual seed ( default = random )") 
    argparser.add_argument("--scheduler_factor", type=int, default=0.5, help="lr scheduler reducing factor when the loss does'nt improve") 
    argparser.add_argument("--scheduler_patience", type=int, default=5, help="lr scheduler patience") 
    argparser.add_argument("--min_lr", type=int, default=10e-6, help="min lr after which the training stops") 
    argparser.add_argument("--threshold", type=float, default=3, help="thresholding of the cost matrix to help the solver") 
    argparser.add_argument("--debug", type=int, default=0, help="verbose level") 
 
    args = argparser.parse_args()    

    device = torch.device(guess_device())

    if args.game_type =="Futoshiki":
        game_utils = Futoshi_utils(train_size = args.train_size, validation_size = args.valid_size,
                                              test_size = args.test_size, batch_size = args.batch_size, path_to_data = args.path_to_data)


    torch.manual_seed(args.seed)

    ### TRAINING ###
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model = train_PLL(args, game_utils, device)
        end.record()

        torch.cuda.synchronize()
        total_time = start.elapsed_time(end) #time in milliseconds

        print("Total training time: " + str(total_time)) 
    else:
        model = train_PLL(args, game_utils, device)

    if args.save_path!="":
        print("Training completed, saving model to disk...")
        PATH = args.save_path
        torch.save(model.state_dict(), PATH)
        #checkpoint = torch.load(PATH, weights_only=True)
        #model.load_state_dict(checkpoint)
        print("model saved to",PATH)

    ### TESTING ###
    test(args, model, game_utils, device)
    print("Done, good bye !")
   
      
if __name__ == "__main__":
    main()
