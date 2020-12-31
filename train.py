import torch
import argparse
from model import VDN,weight_init_kaiming,VDN_2
from torch.utils.data import DataLoader
from loss.loss import loss_fn
import os
from data.data_provider import SingleLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from utils.metric import calculate_psnr
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint
def train(args):
    # torch.set_num_threads(4)
    # torch.manual_seed(args.seed)
    # checkpoint = utility.checkpoint(args)
    data_set = SingleLoader(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory = True
    )

    criterion = loss_fn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    _C = 3
    if args.model_type == "VD":
        model = VDN(_C).to(device)
    elif args.model_type == 'VD_kpn':
        model = VDN_2(_C).to(device)
    else:
        print(" Model type not valid")
        return
    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 25, 30, 35, 40, 45, 50], 0.5)

    optimizer.zero_grad()
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.restart:
        model = weight_init_kaiming(model)
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    else:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', args.load_type)
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            state_dict = checkpoint['state_dict']
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = "model."+ k  # remove `module.`
            #     new_state_dict[name] = v
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            model = weight_init_kaiming(model)
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    for epoch in range(start_epoch, args.epoch):
        for step, data in enumerate(data_loader):
            im_noisy, im_gt, sigmaMapEst, sigmaMapGt = [x.to(device) for x in data]
            # print(im_noisy)
            # print(im_gt)
            # print(sigmaMapEst)
            # print(sigmaMapGt)
            phi_Z, phi_sigma = model(im_noisy,'train')
            # print(pred.size())
            loss, g_lh, kl_g, kl_Igam = criterion(phi_Z, phi_sigma, im_noisy, im_gt,
                                                          sigmaMapGt, 1e-6, radius=3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss.update(loss)
            if global_step % args.save_every == 0:
                print("Save : epoch ",epoch ," step : ", global_step," with avg loss : ",average_loss.get_value() , ",   best loss : ", best_loss )
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False
                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
            if global_step % args.loss_every == 0:
                im_denoise = torch.clamp(im_noisy - phi_Z[:, :_C, ].detach().data, 0.0, 1.0)
                print(global_step, "PSNR  : ", calculate_psnr(im_denoise, im_gt))
                print(average_loss.get_value())
            global_step += 1
        print("Epoch : ", epoch , "end at step: ", global_step)
        scheduler.step()

    # print(model)
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise', help='path to noise folder image')
    parser.add_argument('--gt_dir', '-g' , default='/home/dell/Downloads/gt', help='path to gt folder image')
    parser.add_argument('--image_size', '-sz' , default=64, type=int, help='size of image')
    parser.add_argument('--epoch', '-e' ,default=1000, type=int, help='batch size')
    parser.add_argument('--batch_size','-bs' ,  default=2, type=int, help='batch size')
    parser.add_argument('--save_every','-se' , default=200, type=int, help='save_every')
    parser.add_argument('--loss_every', '-le' , default=1, type=int, help='loss_every')
    parser.add_argument('--restart','-r' ,  action='store_true', help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=2, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint/',
                        help='the checkpoint to eval')
    parser.add_argument('--model_type', '-m' , default="VD", help='type of model : VD, VD_kpn')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')

    args = parser.parse_args()
    #
    train(args)
