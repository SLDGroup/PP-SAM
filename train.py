import os
import time
from datetime import datetime
import random
import argparse
import logging
from collections import defaultdict
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def test(model, path, dataset):
    model.eval()
    predictor_tuned = SamPredictor(model)
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    num1 = len(os.listdir(gt_root))
    images_path_list = [f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    images_path_list = sorted(images_path_list)

    DSC = 0.0
    for i in range(num1):
        image = cv2.imread(image_root+''+images_path_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(gt_root+''+images_path_list[i], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0
        
        H, W = mask.shape
        y_indices, x_indices = np.where(mask > 0)
        
        if(len(x_indices) == 0 or len(y_indices) == 0):
            x_min, x_max = 0, W-1
            y_min, y_max = 0, H-1
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates        
        perturb_h_len = 30 #100
        x_min = max(0, x_min - perturb_h_len)
        x_max = min(W, x_max + perturb_h_len)
        y_min = max(0, y_min - perturb_h_len)
        y_max = min(H, y_max + perturb_h_len)
        input_bbox = np.array([x_min, y_min, x_max, y_max])
        predictor_tuned.set_image(image)

        pred, _, _ = predictor_tuned.predict(
            point_coords=None,
            box=input_bbox,
            multimask_output=False,
        )    
        # eval Dice
        input = np.where(np.array(pred) >= 0.5, 1, 0)
        target = np.where(np.array(mask) > 0.1, 1, 0)
        N = mask.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1

def train(transformed_data, ground_truth_masks, bbox_coords, model, optimizer, epoch, test_path, model_name = 'SAM'):
    model.train()
    global best
    global total_train_time
    time_before_epoch_start = time.time()
    size_rates = [1]

    keys = transformed_data.keys()
    epoch_losses = []
    i = 0
    for k in keys:
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            input_image = transformed_data[k]['image'].cuda()
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']        

            image_embedding = model.image_encoder(input_image)
            
            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device='cuda')
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)
            box_torch = box_torch[None, :]
      
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).cuda()
            
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0.0, dtype=torch.float32)
            upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).cuda()
            
            loss = structure_loss(upscaled_masks, gt_binary_mask)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            i += 1
            if i % 250 == 0 or i == total_step: 
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss.item()))
                           
            epoch_losses.append(loss.item())
    print('EPOCH: '+ str(epoch) + ' loss: ' + str(np.mean(epoch_losses)))
    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    print('total train time till current epoch: '+ str(total_train_time))
    logging.info('total train time till current epoch: '+ str(total_train_time))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '' + model_name + '-last.pth')
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:
    	total_dice = 0
    	total_images = 0
    	for dataset in ['Kvasir']: #['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    	    dataset_dice, n_images = test(model, test_path, dataset)
    	    total_dice += (n_images*dataset_dice)
    	    total_images += n_images
    	    logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
    	    print(dataset, ': ', dataset_dice)
    	    dict_plot[dataset].append(dataset_dice)
    	meandice = total_dice/total_images
    	dict_plot['test'].append(meandice)
    	print('Validation dice score: {}'.format(meandice))
    	logging.info('Validation dice score: {}'.format(meandice))
    	if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            torch.save(model.state_dict(), save_path + '' + model_name + '-best.pth')
    
if __name__ == '__main__':
    dict_plot = {'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-300':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]} #{'CVC-ClinicDB':[], 'test':[]} #
    name = ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test'] #['CVC-ClinicDB', 'test'] 
    n_images = 900
    perturb_h_len = 50
    perturb_l_len = 0
    freeze_image_encoder = 0
    freeze_decoder = 1
    
    ##################model_name#############################
    model_name = 'PolypSAM_freeze_mask_decoder_vit_b_train_p'+str(perturb_l_len)+'_'+str(perturb_h_len)+'_test_p30_Kvasirbest_bs1_random_shot'+str(n_images)+'_e100_Run1' 
    ###############################################
    print(model_name)
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=1, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=1024, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=300, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./data/polyp/TrainDatasetKvasir/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./data/polyp/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log_'+model_name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    model_type = 'vit_b'
    checkpoint = './pretrained_pth/sam/sam_vit_b_01ec64.pth'#sam_vit_b_01ec64.pth' #sam_vit_l_0b3195.pth

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    if freeze_image_encoder:
        print("Freezing image encoder")
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    if freeze_decoder:
        print("Freezing mask decoder")
        for param in model.mask_decoder.parameters():
            param.requires_grad = False
    model.cuda()
    
    # Zero-shot inference of SAM
    total_dice, total_images = 0.0, 0.0
    for dataset in ['Kvasir']: #['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    	dataset_dice, n_imgs = test(model,  opt.test_path, dataset)
    	total_dice += (n_imgs*dataset_dice)
    	total_images += n_imgs
    	logging.info('dataset: {}, dice: {}'.format(dataset, dataset_dice))
    	print(dataset, ': ', dataset_dice)
    meandice = total_dice/total_images
    print('Zero-shot validation dice score: {}'.format(meandice))
    logging.info('Zero-shot validation dice score: {}'.format(meandice))
    
    best = 0

    params = list(model.image_encoder.parameters()) + list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) #+ list(model.out.parameters()) #+ list(model.pvt_cascade.parameters()) #+ list(model.trans2pvt.parameters()) #+ list(model.pvt_stage2.parameters()) + list(model.pvt_norm2.parameters()) + list(model.pvt_stage3.parameters()) + list(model.pvt_norm3.parameters()) + list(model.pvt_stage4.parameters()) + list(model.pvt_norm4.parameters()) + list(model.decoder.parameters()) #.mask_decoder.   

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
        #optimizer = torch.optim.Adam(params, opt.lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)


    # sort images
    images_path_list = sorted([f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
    
    # select n images from the dataset
    img_idxs = random.sample(range(0, len(images_path_list)), n_images)
    print(len(img_idxs), 'image indexes ', img_idxs)
    logging.info(str(len(img_idxs)) + 'image indexes '+ str(img_idxs))

    bbox_coords = {}
    for k in img_idxs:
        im = cv2.imread(gt_root+''+images_path_list[k])
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
        y_indices, x_indices = np.where(gray > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        H, W = gray.shape
        
        ###### For fixed perturbations
        #x_min = max(0, x_min - perturb_h_len) 
        #x_max = min(W, x_max + perturb_h_len) 
        #y_min = max(0, y_min - perturb_h_len) 
        #y_max = min(H, y_max + perturb_h_len)

        ###### For variable perturbations
        x_min = max(0, x_min - np.random.randint(perturb_l_len, perturb_h_len))
        x_max = min(W, x_max + np.random.randint(perturb_l_len, perturb_h_len))
        y_min = max(0, y_min - np.random.randint(perturb_l_len, perturb_h_len))
        y_max = min(H, y_max + np.random.randint(perturb_l_len, perturb_h_len))
        bbox_coords[images_path_list[k]] = np.array([x_min, y_min, x_max, y_max])
        
    #print(bbox_coords)
    transformed_data = defaultdict(dict)
    masks = defaultdict(dict)
    for k in img_idxs:
        image = cv2.imread(image_root+''+images_path_list[k])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(gt_root+''+images_path_list[k], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0
        transform = ResizeLongestSide(model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device='cuda')
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
  
        input_image = model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        transformed_data[images_path_list[k]]['image'] = input_image
        transformed_data[images_path_list[k]]['input_size'] = input_size
        transformed_data[images_path_list[k]]['original_image_size'] = original_image_size
        masks[images_path_list[k]] = mask   

    total_step = len(transformed_data)

    print("#" * 20, "Start Training", "#" * 20)
    total_train_time = 0

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(transformed_data, masks, bbox_coords, model, optimizer, epoch, opt.test_path, model_name = model_name)
    print('avg train time: '+ str(total_train_time/(opt.epoch-1)))
    logging.info('avg train time: '+ str(total_train_time/(opt.epoch-1)))
