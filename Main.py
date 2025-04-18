"""
Implementation of "Denoising Diffusion Error Correction Codes" (DDECC), in ICLR23
https://arxiv.org/abs/2209.13533
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from DDECC import DDECCT


##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

##################################################################


class FEC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        std_noise = random.choice(self.sigma)
        z = torch.randn(self.code.n) * std_noise
        #h = torch.from_numpy(np.random.rayleigh(1,self.code.n)).float()
        h=1
        y = h*bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()#, torch.tensor([std_noise]).float()


##################################################################
##################################################################

def train(model, device, train_loader, optimizer, epoch, LR): # loops through batches of training data
    model.train()
    cum_loss = cum_samples = 0
    t = time.time()
    for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(
            train_loader):
        loss = model.loss(bin_to_sign(x)) # calls model.loss() to compute the loss
        model.zero_grad()
        loss.backward() # performs backpropagationand updates the weights
        optimizer.step()
        model.ema.update(model) # uses an exponential moving average (EMA) for model stabilization 
        ###
        cum_loss += loss.item() * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx+1) % 500 == 0 or batch_idx == len(train_loader) - 1: # logs training progress every 500 iterations 
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples

##################################################################

def test(model, device, test_loader_list, EbNo_range_test, min_FER=100, max_cum_count=1e7, min_cum_count=1e5):
    model.eval()
    test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_fer = cum_count = 0.
            _, x_pred_list, _, _ = model.p_sample_loop(next(iter(test_loader))[3])
            test_ber_ddpm , test_fer_ddpm = [0]*len(x_pred_list), [0]*len(x_pred_list)
            idx_conv_all = []
            while True:
                (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                x_pred, x_pred_list, idx_conv,synd_all = model.p_sample_loop(y)
                x_pred = sign_to_bin(torch.sign(x_pred))

                idx_conv_all.append(idx_conv)
                for kk, x_pred_tmp in enumerate(x_pred_list):
                    x_pred_tmp = sign_to_bin(torch.sign(x_pred_tmp))

                    test_ber_ddpm[kk] += BER(x_pred_tmp, x) * x.shape[0]
                    test_fer_ddpm[kk] += FER(x_pred_tmp, x) * x.shape[0]
                    
                test_ber += BER(x_pred, x) * x.shape[0]
                test_fer += FER(x_pred, x) * x.shape[0]
                cum_count += x.shape[0]
                if (min_FER > 0 and test_fer > min_FER and cum_count > min_cum_count) or cum_count >= max_cum_count:
                    if cum_count >= 1e9:
                        logging.info(f'Cum count reached EbN0:{EbNo_range_test[ii]}')
                    else:    
                        logging.info(f'FER count treshold reached EbN0:{EbNo_range_test[ii]}')
                    break
            idx_conv_all = torch.stack(idx_conv_all).float()
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            for kk in range(len(test_ber_ddpm)):
                test_ber_ddpm[kk] /= cum_count
                test_fer_ddpm[kk] /= cum_count
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER_DDPM={test_ber_ddpm}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, -ln(BER)_DDPM={[-np.log(elem) for elem in test_ber_ddpm]}')
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, FER_DDPM={test_fer_ddpm}')
            logging.info(f'#It. to zero syndrome: Mean={idx_conv_all.mean()}, Std={idx_conv_all.std()}, Min={idx_conv_all.min()}, Max={idx_conv_all.max()}')
        ###
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_fer_list, EbNo_range_test))]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, -np.log(elem)) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ber_list, test_loss_fer_list
##################################################################
##################################################################
##################################################################


def main(args): # Main Training Loop
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    model = DDECCT(args, device=device,dropout=0).to(device) # initializes the model (DDECCT)
    model.ema.register(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # set up the optimizer (Adam)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6) # sets up the learning rate scheduler (CosineAnnealingLR)

    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    # Prepare the training and testing datasets with different noise levels (EbNo values)
    EbNo_range_test = range(4, 7)
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    train_dataloader = DataLoader(FEC_Dataset(code, std_train, len=args.batch_size * 1000, zero_cw=True), batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(FEC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(std_test))]
    #################################
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1): # runs training for specific number of epochs
        loss= train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model_U_LDPC_49_24')) # saves the best performing model
            logging.info(f'Model Saved')
        #if epoch % (args.epochs//2) == 0 or epoch in [1,25]:
            #test(model, device, test_dataloader_list, EbNo_range_test,min_FER=50,max_cum_count=1e6,min_cum_count=1e4)
    ############
    ############
    # At the end of training: 
    logging.info('Loading Best Model') # loads the best model and evaluates it on test data
    model = torch.load(os.path.join(args.path, 'best_model_U_LDPC_49_24')).to(device)
    logging.info('Regular Reverse Diffusion')
    test(model, device, test_dataloader_list, EbNo_range_test,min_FER=100 ,max_cum_count=1e6,min_cum_count=1e4)
    logging.info('Line Search Reverse Diffusion') # performs Line Search Reverse Diffusion for enhanced decoding
    model.line_search = True
    test(model, device, test_dataloader_list, EbNo_range_test,min_FER=100)

##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDPM_ECCT')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpus', type=str, default='1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128) #128 # Training Batch size
    parser.add_argument('--test_batch_size', type=int, default=2048) #2048 # Testing Batch size
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='LDPC',
                        choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=24)
    parser.add_argument('--code_n', type=int, default=49)

    # model args
    parser.add_argument('--N_dec', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--h', type=int, default=8)

    # DDECC args
    parser.add_argument('--sigma', type=float, default=0.01)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    class Code():
        pass
    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    G, H = Get_Generator_and_Parity(code)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    ###
    args.N_steps = code.pc_matrix.shape[0]+5 #number of diffusion steps
    ####################################################################
    model_dir = os.path.join('DDECCT_Results',
                             args.code_type + '__Code_n_' + str(
                                 args.code_n) + '_k_' + str(
                                 args.code_k) + '__' + datetime.now().strftime(
                                 "%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    main(args) # call main(args) to start training
