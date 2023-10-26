import torch
from math import ceil
from copy import deepcopy
from tqdm import tqdm


# Statistical Recurrent Unit class (based on paper by Junier B. Oliva, arXiv:1703.00381v1)
class eSRU_2LF(torch.nn.Module):
    
    def __init__(self, 
                 n_inp_channels,  # dimension of input sequence 
                 n_out_channels,  # dimension of output (predicted) sequence
                 dim_iid_stats,   # dimension of iid statistics \phi
                 dim_rec_stats,   # dimension of recurrent stats u
                 dim_rec_stats_feedback, # dimension of recurrent starts fed back as 'r' to generate iid stats 
                 dim_final_stats, # dimension of final stats u
                 A,               # Set of scales for exponentially weighted moving averages
                 device           # CPU/GPU memory for storing tensors
                ):

        # inherit the default attributes of Module class
        super(eSRU_2LF, self).__init__()
        
        # initialization of SRU parameters
        self.type            = 'eSRU_2LF'
        self.n_inp_channels  = n_inp_channels  # dimension of input data
        self.n_out_channels  = n_out_channels  # dimension of predicted output
        self.dim_iid_stats   = dim_iid_stats   # dimension of 'phi_t'
        self.dim_rec_stats   = dim_rec_stats   # dimension of 'u_t'
        self.dim_final_stats = dim_final_stats # dimension of 'o_t'
        self.dim_rec_stats_feedback = dim_rec_stats_feedback # dimension of 'r_t'
        self.numScales       = len(A)

        # Take kroneck product: A \otimes 1_{dim_iid_stats}       
        self.A_mask = torch.Tensor([x for x in(A) for i in range(dim_iid_stats)]).view(1, -1)
        self.A_mask.requires_grad = False
        self.A_mask = self.A_mask.to(device) # shift to GPU memory

        # Initialization of SRU cell's tensors
        self.phi_t = torch.zeros(dim_iid_stats,1, requires_grad=True, device=device)
        self.phi_tile = torch.zeros(dim_iid_stats*self.numScales,1, requires_grad=True, device=device)
        self.r_t   = torch.zeros(dim_rec_stats_feedback,1, requires_grad=True, device=device)
        self.o_t   = torch.zeros(dim_final_stats,1, requires_grad=True, device=device)
        self.y_t   = torch.zeros(n_out_channels,1, requires_grad=True, device=device)
        self.u_t   = torch.zeros(1, dim_rec_stats * self.numScales, requires_grad=True, device=device)
        self.u_t_prev   = torch.zeros(1, dim_rec_stats * self.numScales, device=device)        
        
        # MLPs in SRU cell
        self.lin_xr2phi = torch.nn.Linear(n_inp_channels + dim_rec_stats_feedback, dim_iid_stats, bias=True)
        self.lin_r1 = torch.nn.Linear(dim_rec_stats_feedback, dim_rec_stats_feedback, bias=True)
        self.lin_r2 = torch.nn.Linear(dim_rec_stats_feedback, dim_rec_stats_feedback, bias=True)
        self.lin_o = torch.nn.Linear(self.numScales*dim_rec_stats, dim_final_stats, bias=True)
        self.lin_y = torch.nn.Linear(dim_final_stats, n_out_channels, bias=True)

        # Fixed random matrices for sketching hidden state to lower dimensions
        self.intrMat_h2r_transpose = (dim_rec_stats_feedback ** (-0.5) *
            torch.randn(self.numScales*dim_rec_stats, dim_rec_stats_feedback, requires_grad=False, device=device))
    
    # SRU forward pass     
    def forward(self, x_t):   

        # Generate feedback statistics 
        self.r_t = torch.matmul(self.u_t_prev, self.intrMat_h2r_transpose) # sketch of hidden state
        self.r_t = torch.nn.functional.elu(self.lin_r1(self.r_t)) # layer 1
        self.r_t = torch.nn.functional.elu(self.lin_r2(self.r_t)) # layer 2

        # Generate iid statistics: phi_t
        self.phi_t = torch.nn.functional.elu(self.lin_xr2phi(torch.cat((x_t, torch.flatten(self.r_t)))))
        
        # Generate multiscale recurrent statistics: u_t
        self.phi_tile = self.phi_t.repeat(1, self.numScales)
        self.u_t = torch.mul(self.A_mask, self.u_t_prev) + torch.mul((1-self.A_mask), self.phi_tile)
        self.u_t_prev.data = self.u_t.data

        # Generate final statistics: o_t
        self.o_t = torch.nn.functional.elu(self.lin_o(self.u_t))
        
        # Generate predicted output: y_t
        self.y_t = self.lin_y(self.o_t)

        return self.y_t
        
        
    def reset_recurrent_stats(self):
        self.u_t_prev.fill_(0)

############################################
# trainSRU_eSRU_2LF
############################################
def train_eSRU_2LF(model, X, device, batch_size, predicted_idx, max_iter,
                   lambda1, lambda2, lambda3, lr, lr_gamma, lr_update_gap, stopping_thresh):

    stoppingCntr = 0
    stoppingCntrThr = 10
    n = X.shape[0]
    wtMtxRow = torch.zeros(model.numScales * model.dim_final_stats, 1, requires_grad = False, device=device)

    lin_xr2phi_weight = deepcopy(model.lin_xr2phi.weight.data)
    lin_xr2phi_bias = deepcopy(model.lin_xr2phi.bias.data)
    lin_r1_weight = deepcopy(model.lin_r1.weight.data)
    lin_r1_bias = deepcopy(model.lin_r1.bias.data)
    lin_r2_weight = deepcopy(model.lin_r2.weight.data)
    lin_r2_bias = deepcopy(model.lin_r2.bias.data)
    lin_o_weight = deepcopy(model.lin_o.weight.data)
    lin_o_bias = deepcopy(model.lin_o.bias.data)
    lin_y_weight = deepcopy(model.lin_y.weight.data)
    lin_y_bias = deepcopy(model.lin_y.bias.data)

    #####################################
    # Initialize miscellaneous tensors
    #####################################
    lossVec = torch.zeros(max_iter,2)
    lossVec.to(device)

    mseLoss = torch.nn.MSELoss(reduction = 'sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_update_gap, lr_gamma)

    batch_count = 0
    training_loss = 0
    fit_err = 0
    optimizer.zero_grad()
    num_batches = ceil(X.shape[1] / batch_size)

    for epoch in tqdm(range(max_iter), desc=f'Node {predicted_idx+1}/{X.shape[0]}'):
        # Make deep copy of trainable model parameters to use later in checking the stopping criterion
        with torch.no_grad():
            lin_xr2phi_weight[:,:] = model.lin_xr2phi.weight.data[:,:]
            lin_xr2phi_bias[:] = model.lin_xr2phi.bias.data[:]
            lin_r1_weight[:,:] = model.lin_r1.weight.data[:,:]
            lin_r1_bias[:] = model.lin_r1.bias.data[:]
            lin_r2_weight[:,:] = model.lin_r2.weight.data[:,:]
            lin_r2_bias[:] = model.lin_r2.bias.data[:]
            lin_o_weight[:,:] = model.lin_o.weight.data[:,:]
            lin_o_bias[:] = model.lin_o.bias.data[:]
            lin_y_weight[:,:] = model.lin_y.weight.data[:,:]
            lin_y_bias[:] = model.lin_y.bias.data[:]

        # Update start and stop times for next training batch
        batch_count = batch_count + 1
        if batch_count == num_batches:
            batch_count = 0
            training_loss = 0
            fit_err = 0
        start_time = batch_count * batch_size
        stop_time = min((batch_count + 1) * batch_size, X.shape[1])
        inv_batch_size_m1 = 1 / (stop_time - start_time - 1)

        # Reset recurrent stats u_t
        optimizer.zero_grad()
        model.reset_recurrent_stats()
        
        # Forward pass
        smooth_loss_list = []
        for tt in range(start_time, stop_time - 1):
            model.forward(X[:, tt])
            smooth_loss = inv_batch_size_m1 * mseLoss(
                torch.flatten(model.y_t),
                torch.unsqueeze(X[predicted_idx, tt + 1], 0)
            )
            smooth_loss_list.append(smooth_loss)

        # Use autograd to compute the backward pass (accumulate gradients on each pass).
        model.lin_xr2phi.weight.retain_grad()
        sum(smooth_loss_list).backward()
        # sum([smooth_loss_list[i] for i in range(batch_size_m1)]).backward()
        lossVec[epoch][0] = sum(x.item() for x in smooth_loss_list)
        
        optimizer.step()
        optimizer.zero_grad()

        #Adjust for regularization
        lr_current = optimizer.param_groups[0]['lr']
        softshrink1 = torch.nn.Softshrink(lambda1*lr)
        softshrink2 = torch.nn.Softshrink(lambda2*lr)
        softshrink3 = torch.nn.Softshrink(lambda3*lr)
        with torch.no_grad():
            
            # Update all network parameters except for input layer weight matrix
            model.lin_xr2phi.weight[:,n:].data = softshrink1(model.lin_xr2phi.weight[:,n:]).data
            model.lin_xr2phi.bias.data   = softshrink1(model.lin_xr2phi.bias).data
            model.lin_r1.weight.data = softshrink1(model.lin_r1.weight).data
            model.lin_r1.bias.data   = softshrink1(model.lin_r1.bias).data
            model.lin_r2.weight.data = softshrink1(model.lin_r2.weight).data
            model.lin_r2.bias.data   = softshrink1(model.lin_r2.bias).data
            model.lin_o.bias.data   = softshrink1(model.lin_o.bias).data
            model.lin_y.weight.data = softshrink1(model.lin_y.weight).data
            model.lin_y.bias.data   = softshrink1(model.lin_y.bias).data
            
            # Update input layer weight matrix
            inpWgtMtx = model.lin_xr2phi.weight[:,:n]
            l2normTensor = torch.norm(inpWgtMtx, p=2, dim=0, keepdim=True) # 1 x n row tensor
            model.lin_xr2phi.weight.data[:,:n] = inpWgtMtx*(softshrink2(l2normTensor)/torch.clamp(l2normTensor, min=lambda2*lr_current*0.1))

            # Update the weight matrix mapping multi-time scale hidden state to 
            # the lag sensitive features for prediction purpose
            for rr in range(model.dim_final_stats):
                wtMtxRow.data = model.lin_o.weight.data[rr,:]
                #reshape wtMtxRow as (numScales x dim_rec_stats) matrix
                wtMtxRowReshaped = wtMtxRow.view(model.numScales, model.dim_rec_stats)
                l2normTensor1 = torch.norm(wtMtxRowReshaped, p=2, dim=0, keepdim=True) # 1 x dim_final_stats row tensor
                model.lin_o.weight.data[rr,:] = (wtMtxRowReshaped*(softshrink3(l2normTensor1)/torch.clamp(l2normTensor1, min=lambda3*lr_current*0.1))).flatten().data[:]

            # Compute and log regularization loss without updating gradients
            loss1 = lambda1*((torch.norm(model.lin_y.weight.data, 1)+ torch.norm(model.lin_y.bias.data, 1) + 
                    torch.norm(model.lin_xr2phi.weight[:,n:].data, 1)) + torch.norm(model.lin_xr2phi.bias.data, 1) + 
                    torch.norm(model.lin_o.weight.data, 1) + torch.norm(model.lin_o.bias.data, 1) +
                    torch.norm(model.lin_r1.weight.data, 1) + torch.norm(model.lin_r1.bias.data, 1) + 
                    torch.norm(model.lin_r2.weight.data, 1) + torch.norm(model.lin_r2.bias.data, 1)) 
            lossVec[epoch][1] = lossVec[epoch][1] + loss1.item()
            loss2 = lambda2*torch.sum(torch.norm(model.lin_xr2phi.weight.data, p=2, dim=0)[:n])
            lossVec[epoch][1] = lossVec[epoch][1] + loss2.item()

            # Again force gradient to be zero (just to be extra safe)
            optimizer.zero_grad()
            scheduler.step()


        # Record total-loss for current epoch
        lossVec[epoch][1] = lossVec[epoch][1] + lossVec[epoch][0]
        training_loss = training_loss + lossVec[epoch][1] 
        fit_err = fit_err + lossVec[epoch][0]    

        with torch.no_grad():
            paramDelta = (mseLoss(model.lin_y.weight, lin_y_weight) 
                      + mseLoss(model.lin_y.bias, lin_y_bias) 
                      + mseLoss(model.lin_xr2phi.weight, lin_xr2phi_weight) 
                      + mseLoss(model.lin_xr2phi.bias, lin_xr2phi_bias) 
                      + mseLoss(model.lin_o.weight, lin_o_weight)
                      + mseLoss(model.lin_o.bias, lin_o_bias) 
                      + mseLoss(model.lin_r1.weight, lin_r1_weight) 
                      + mseLoss(model.lin_r1.bias, lin_r1_bias)
                      + mseLoss(model.lin_r2.weight, lin_r2_weight) 
                      + mseLoss(model.lin_r2.bias, lin_r2_bias)).data

        # Stopping criterion 
        if(paramDelta < stopping_thresh):
            stoppingCntr = stoppingCntr + 1
            if(stoppingCntr == stoppingCntrThr):
                break
        else:
            stoppingCntr = 0

    return model, lossVec
