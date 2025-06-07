import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths
import pdb
from torchvision.transforms import (Compose, ToPILImage, ToTensor,
                                    RandAugment, RandomCrop,
                                    RandomHorizontalFlip, RandomErasing,ColorJitter, GaussianBlur)
NUM_CLASSES = len(my_bidict)

def classifier(model, data_loader, device):
    model.eval()
    acc = ratio_tracker()                  
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = fast_predict(model, x, NUM_CLASSES)
            acc.update((pred == y).sum().item(), y.size(0))
    return acc.get_ratio()

def fast_predict(model, x, num_classes):
    """
    Bayes‑rule
    x : (B,C,H,W)  Tensor  (cuda)
    return : (B,)  LongTensor
    """
    B = x.size(0)
    ll_list = []
    with torch.no_grad():
        for lbl in range(num_classes):
            lbl_vec = torch.full((B,), lbl, dtype=torch.long, device=x.device)
            pix_out = model(x, lbl_vec)           # (B, ⋯)
            ll      = - discretized_mix_logistic_loss(x, pix_out, Bayes=True)
            ll_list.append(ll.view(-1,1))
    ll_all = torch.cat(ll_list, dim=1)            # (B, K)
    return ll_all.argmax(1)                       # (B,)

def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode = 'training', ema=None):
    if mode == 'training':
        model.train()
    else:
        model.eval()
        
    deno =  args.batch_size * np.prod(args.obs) * np.log(2.)        
    loss_tracker = mean_tracker()
    acc_tracker = ratio_tracker() 
    
    for batch_idx, (model_input,labels) in enumerate(tqdm(data_loader)):
        model_input = model_input.to(device) #model_input.shape torch.Size([64, 3, 32, 32])
        labels = labels.to(device)
        model_output = model(model_input,labels) #model_output.shape torch.Size([64, 50, 32, 32])
        preds = fast_predict(model, model_input, NUM_CLASSES)
        acc_tracker.update((preds == labels).sum().item(), labels.size(0))

        # pdb.set_trace()
        # sum over discretized_mix_logistic_loss
        if mode == 'training':
            loss = loss_op(model_input, model_output, Bayes=False)
        else:
            loss = loss_op(model_input, model_output,Bayes=True)

        loss_tracker.update(loss.item()/deno)
        if mode == 'training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:          
                ema.update(model)
        
    # if args.en_wandb:
    #     wandb.log({mode + "-Average-BPD" : loss_tracker.get_mean()})
    #     wandb.log({mode + "-epoch": epoch})
    #     wandb.log({mode + "Training accuracy" : acc_tracker.get_ratio()})"
    if args.en_wandb:
        wandb.log({
        f"{mode}-Average-BPD": loss_tracker.get_mean(),
        f"{mode}/acc"       : acc_tracker.get_ratio(),   
        "epoch"             : epoch
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-w', '--en_wandb', type=bool, default=False,
                            help='Enable wandb logging')
    parser.add_argument('-t', '--tag', type=str, default='default',
                            help='Tag for this run')
    
    # sampling
    parser.add_argument('-c', '--sampling_interval', type=int, default=5,
                        help='sampling interval')
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-sd', '--sample_dir',  type=str, default='samples',
                        help='Location for saving samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cpen455', help='Can be either cifar|mnist|cpen455')
    parser.add_argument('-st', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--obs', type=tuple, default=(3, 32, 32),
                        help='Observation shape')
    
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=1,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=40,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-sb', '--sample_batch_size', type=int, default=32,
                        help='Batch size during sampling per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    check_dir_and_create(args.save_dir)
    
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model_name = 'pcnn_' + args.dataset + "_"
    model_path = args.save_dir + '/'
    if args.load_params is not None:
        model_name = model_name + 'load_model'
        model_path = model_path + model_name + '/'
    else:
        model_name = model_name + 'from_scratch'
        model_path = model_path + model_name + '/'
    
    job_name = "PCNN_Training_" + "dataset:" + args.dataset + "_" + args.tag
    
    if args.en_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set entity to specify your username or team name
            # entity="qihangz-work",
            # set the wandb project where this run will be logged
            project="CPEN455HW",
            # group=Group Name
            name=job_name,
        )
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wandb.config.update(args)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Reminder: if you have patience to read code line by line, you should notice this comment. here is the reason why we set num_workers to 0:
    #In order to avoid pickling errors with the dataset on different machines, we set num_workers to 0.
    #If you are using ubuntu/linux/colab, and find that loading data is too slow, you can set num_workers to 1 or even bigger.
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':True}

    # set data
    if "mnist" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), rescaling, replicate_color_channel])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                            train=True, transform=ds_transforms), batch_size=args.batch_size, 
                                shuffle=True, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    elif "cifar" in args.dataset:
        ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
        if args.dataset == "cifar10":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        elif args.dataset == "cifar100":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            raise Exception('{} dataset not in {cifar10, cifar100}'.format(args.dataset))
    
    elif "cpen455" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling]) #original
        # cutout_scale = (0.24, 0.26)      # (min, max)  ← 길이 2짜리 튜플!

        # train_transforms = Compose([
        #     ToPILImage(),
        #     RandomCrop(32, padding=4, padding_mode='reflect'),
        #     RandomHorizontalFlip(),
        #     ColorJitter(0.2, 0.2, 0.2, 0.1),         
        #     RandAugment(num_ops=2, magnitude=9), 
        #     ToTensor(),     
        #     RandomErasing(p=0.5, scale=(0.1, 0.25)), 
        #     rescaling
        # ])

        val_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        rescaling
        ])

        train_loader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'train', 
                                                                  transform=ds_transforms), #ds_transform to train_transforms
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
        test_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'test', 
                                                                  transform=ds_transforms), #ds_transform to val_transforms
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
        val_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'validation', 
                                                                  transform=ds_transforms), #ds_transform to val_transforms
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
    else:
        raise Exception('{} dataset not in {mnist, cifar, cpen455}'.format(args.dataset))
    
    args.obs = (3, 32, 32)
    input_channels = args.obs[0]
    
    def loss_op(real, fake, Bayes=False):
        return discretized_mix_logistic_loss(real, fake, Bayes=False)

    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix,film=True,late_fusion=True, mid_fusion=True)
    model = model.to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    ema = EMA(model, decay=0.999)

    
    for epoch in tqdm(range(args.max_epochs)):
        train_or_test(model = model, 
                      data_loader = train_loader, 
                      optimizer = optimizer, 
                      loss_op = loss_op, 
                      device = device, 
                      args = args, 
                      epoch = epoch, 
                      mode = 'training',
                      ema = ema)
        
        # decrease learning rate
        scheduler.step()
        # train_or_test(model = model,
        #               data_loader = test_loader,
        #               optimizer = optimizer,
        #               loss_op = loss_op,
        #               device = device,
        #               args = args,
        #               epoch = epoch,
        #               mode = 'test')
        
        train_or_test(model = model,
                      data_loader = val_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'val')
        if epoch % 5 == 0:                 # 매 5 epoch 마다
            val_acc = classifier(model, val_loader, device)
            print(f"[epoch {epoch}]  val_acc = {val_acc:.3f}")
            if args.en_wandb:
                wandb.log({"val/acc": val_acc, "epoch": epoch})
        
        if epoch % args.sampling_interval == 0:
            print('......sampling......')

            ema.copy_to(model)

            class_labels = torch.tensor([0, 1, 2, 3] * (args.sample_batch_size // 4), device=device)
            sample_t = sample(model, args.sample_batch_size, args.obs, sample_op,class_labels) #conditional
            sample_t = rescaling_inv(sample_t)
            save_images(sample_t, args.sample_dir)
            sample_result = wandb.Image(sample_t, caption="epoch {}".format(epoch))
            
            gen_data_dir = args.sample_dir
            ref_data_dir = args.data_dir +'/test'
            paths = [gen_data_dir, ref_data_dir]
            try:
                fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
                print("Dimension {:d} works! fid score: {}".format(192, fid_score))
            except:
                print("Dimension {:d} fails!".format(192))
                
            if args.en_wandb:
                wandb.log({"samples": sample_result,
                            "FID": fid_score})
            ema.restore(model)
        
        if (epoch + 1) % args.save_interval == 0: 
            # if not os.path.exists("models"):
            #     os.makedirs("models")
            local_dir = './models' 
            os.makedirs(local_dir, exist_ok=True)

            CKPT_DIR = './content/drive/MyDrive/CPEN455'
            os.makedirs(CKPT_DIR, exist_ok=True)


            ema.copy_to(model)

            # torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
            torch.save(model.state_dict(), f'{local_dir}/{model_name}_{epoch+1}.pth')
            torch.save(model.state_dict(), f'{CKPT_DIR}/epoch_{epoch+1}.pth')

            torch.save({'epoch': epoch+1,
                'model': model.state_dict(),
                'optim': optimizer.state_dict()},
               f'{CKPT_DIR}/pcnn_e{epoch+1}.pth')
               
            ema.restore(model)

    save_name = f'models/conditional_pixelcnn_{args.tag}.pth'
    torch.save(model.state_dict(), save_name)
    print(f"Saved model: {save_name}")        
