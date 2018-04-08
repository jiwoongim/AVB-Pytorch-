import argparse, os, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import utils 
from AVB import AVB

def train(model, args, data_loader):


    print('---------- Networks architecture -------------')
    utils.print_network(model.disc_net)
    utils.print_network(model.encoder)
    utils.print_network(model.decoder)
    print('-----------------------------------------------')

    disc_optimizer = optim.Adam(model.disc_net.parameters(), lr=args.lrDisc, betas=(args.beta1, args.beta2))
    enc_optimizer = optim.Adam(model.encoder.parameters(), lr=args.lrEnc, betas=(args.beta1, args.beta2))
    dec_optimizer = optim.Adam(model.decoder.parameters(), lr=args.lrDec, betas=(args.beta1, args.beta2))


    train_hist = {}
    train_hist['tr_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    model.disc_net.train()
    model.encoder.train()
    model.decoder.train()
    print('training start!!')
    start_time = time.time()
    for epoch in range(args.epoch):

        epoch_start_time = time.time()
        for iter, (x_, y_) in enumerate(data_loader):
            if iter * args.batch_size < 50000:
                if iter == data_loader.dataset.__len__() // args.batch_size:
                    break
                
                N, C, IW, IH = x_.size()
                z_ = torch.randn((args.batch_size, args.z_dim))
                if args.gpu_mode:
                    x_ = Variable(x_.cuda())
                    z_ = Variable(z_.cuda())
                else:
                    x_ = Variable(x_)
                    z_ = Variable(z_)

                #if epoch < 150 or (epoch >= 150 and epoch %2 == 0):
                #if np.random.randint(2, size=1)[0]:
                for i in range(1):
                    z_x_= model.encoder(x_)
                    z_x_= torch.autograd.Variable(z_x_.data, requires_grad=False)

                    # update Discriminator network 
                    lrDisc = max([args.lrDisc / (1.0 + epoch / 50.0), 0.000001])
                    disc_optimizer.zero_grad()
                    lossD = model.disc_net.loss(x_, z_x_, z_)
                    lossD.backward()
                    #for g in disc_optimizer.param_groups: g['lr'] = lrDisc
                    disc_optimizer.step() 


                # update Encoder & Decoder network
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                lrEnc = max([args.lrEnc / (1.0 + epoch / 100.0), 0.000001])
                lrDec = max([args.lrDec / (1.0 + epoch / 100.0), 0.000001])
                #for g in enc_optimizer.param_groups: g['lr'] = lrEnc
                #for g in dec_optimizer.param_groups: g['lr'] = lrDec
                beta = min([float(epoch) / args.anneal_steps, 1.0])
                loss = model.loss(x_, beta) 
                loss.backward()
                train_hist['tr_loss'].append(loss.data[0])


                if np.isnan(loss.data.cpu().numpy()):
                    print('loss AVB nan')
                    import pdb; pdb.set_trace()

                # `clip_grad_norm` helps prevent the exploding gradient problem.
                torch.nn.utils.clip_grad_norm(model.encoder.parameters(), args.clip)
                enc_optimizer.step()
                dec_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] Beta %.2f, lossV: %.8f | lrD %.5f, lossD: %.8f" %
                            ((epoch + 1), \
                            (iter + 1), \
                            len(data_loader.dataset) // args.batch_size, \
                            beta, loss.data[0] *784, lrDisc, lossD.data[0]))

        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        visualize_results(model, epoch+1, args)

        if epoch % 25 :
            save(model, epoch, args.save_dir, args.dataset, \
                    args.model_type, args.batch_size, train_hist)

    train_hist['total_time'].append(time.time() - start_time)
    print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % \
                    (np.mean(train_hist['per_epoch_time']),
                    epoch, train_hist['total_time'][0]))
    print("Training finish!... save training results")

    save(model, epoch, args.save_dir, args.dataset, args.model_type, \
                                    args.batch_size, train_hist)
    utils.generate_animation(args.result_dir + '/' + args.dataset + '/' \
                    + args.model_type + '/' + args.model_type, epoch)
    utils.loss_plot(train_hist, os.path.join(args.save_dir, args.dataset, \
                                    args.model_type), args.model_type)


def visualize_results(model, epoch, args, sample_num=100, fix=True):

    model.disc_net.eval()
    model.encoder.eval()
    model.decoder.eval()

    if not os.path.exists(args.result_dir + '/' + args.dataset + '/' + args.model_type):
        os.makedirs(args.result_dir + '/' + args.dataset + '/' + args.model_type)

    tot_num_samples = min(sample_num, args.batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    if fix:
        """ fixed noise """
        samples = model.decoder(model.sample_z_)
    else:
        """ random noise """
        if args.gpu_mode:
            sample_z_ = Variable(torch.rand((args.batch_size, 1, args.z_dim)).cuda(), volatile=True)
        else:
            sample_z_ = Variable(torch.rand((args.batch_size, 1, args.z_dim)), volatile=True)

        samples = model.sample(sample_z_)

    #N,C,IW,IH = samples.size()
    #samples = samples.view([N,C,IW,IH])
    if args.gpu_mode:
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
    else:
        samples = samples.data.numpy().transpose(0, 2, 3, 1)

    utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      args.result_dir + '/' + args.dataset + '/' + args.model_type + '/' + args.model_type + '_epoch%03d' % epoch + '.png')


def save(model, epoch, save_dir, dataset, model_type, batch_size, train_hist):
    save_dir = os.path.join(save_dir, dataset, model_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.encoder.state_dict(), os.path.join(save_dir, model_type + '_encoder_epoch' + str(epoch)+'_batch_sz' + str(batch_size)+'_encoder.pkl'))
    torch.save(model.decoder.state_dict(), os.path.join(save_dir, model_type + '_encoder_epoch' + str(epoch)+'_batch_sz' + str(batch_size)+'_decoder.pkl'))
    torch.save(model.disc_net.state_dict(), os.path.join(save_dir, model_type + '_encoder_epoch' + str(epoch)+'_batch_sz' + str(batch_size)+'disc_net.pkl'))

    with open(os.path.join(save_dir, model_type + '_history.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)


def load(model, save_dir, dataset='MNIST', model_type='AVB'):

    save_dir = os.path.join(save_dir, dataset, model_type)
    model.encoder.load_state_dict(torch.load(os.path.join(save_dir, model_type + '_encoder.pkl')))
    model.decoder.load_state_dict(torch.load(os.path.join(save_dir, model_type + '_decoder.pkl')))
    model.disc_net.load_state_dict(torch.load(os.path.join(save_dir, model_type + '_disc_net.pkl')))




"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of AVB collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_type', type=str, default='AVB',
                        choices=['AVB', 'DAVB'],
                        help='The type of AVB')#, required=True)
    parser.add_argument('--arch_type', type=str, default='fc',
                        choices=['conv', 'fc'],
                        help='The Architecture Type')#, required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--acF', type=bool, default=False)
    parser.add_argument('--z_dim', type=float, default=32)
    parser.add_argument('--dim_sam', type=float, default=10)
    parser.add_argument('--lrDisc', type=float, default=1e-4)
    parser.add_argument('--lrDec', type=float, default=1e-5)
    parser.add_argument('--lrEnc', type=float, default=1e-5)
    parser.add_argument('--anneal_steps', type=float, default=1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--gpu_mode', type=bool, default=True)

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()


    # declare instance for AVB
    if args.model_type == 'AVB' or args.model_type == 'DAVB' :
        model = AVB(args)
    else:
        raise Exception("[!] There is no option for " + args.model_type)


    # load dataset
    data_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=True,
                                                        transform=transforms.Compose(
                                                            [transforms.ToTensor()])),
                                            batch_size=args.batch_size, shuffle=False)


    # launch the graph in a session
    train(model, args, data_loader)
    print(" [*] Training finished!")

    # visualize learned generator
    model.visualize_results(args.epoch)
    print(" [*] Testing finished!")




if __name__ == '__main__':
    main()

