# Semantic Segmentation
# Code by GunhoChoi

from FusionNet import * 
from UNet import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
parser.add_argument("--batch_size",type=int,default=1,help="batch size")
parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
args = parser.parse_args()

# hyperparameters

batch_size = args.batch_size
img_size = 256
lr = 0.0002
epoch = 100

# input pipeline

img_dir = "./maps/"
img_data = dset.ImageFolder(root=img_dir, transform = transforms.Compose([
                                            transforms.Scale(size=img_size),
                                            transforms.CenterCrop(size=(img_size,img_size*2)),
                                            transforms.ToTensor(),
                                            ]))
img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)

# initiate Generator

if args.network == "fusionnet":
	generator = nn.DataParallel(FusionGenerator(3,3,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
elif args.network == "unet":
	generator = nn.DataParallel(UnetGenerator(3,3,64),device_ids=[i for i in range(args.num_gpu)]).cuda()

# load pretrained model

try:
    generator = torch.load('./model/{}.pkl'.format(args.network))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# loss function & optimizer

recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)

# training

file = open('./{}_mse_loss'.format(args.network), 'w')
for i in range(epoch):
    for _,(image,label) in enumerate(img_batch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3) 
        
        gen_optimizer.zero_grad()

        x = Variable(satel_image).cuda(0)
        y_ = Variable(map_image).cuda(0)
        y = generator.forward(x)
        
        loss = recon_loss_func(y,y_)
        file.write(str(loss)+"\n")
        loss.backward()
        gen_optimizer.step()

        if _ % 400 ==0:
            print(i)
            print(loss)
            v_utils.save_image(x.cpu().data,"./result/original_image_{}_{}.png".format(i,_))
            v_utils.save_image(y_.cpu().data,"./result/label_image_{}_{}.png".format(i,_))
            v_utils.save_image(y.cpu().data,"./result/gen_image_{}_{}.png".format(i,_))
            torch.save(generator,'./model/{}.pkl'.format(args.network))    
