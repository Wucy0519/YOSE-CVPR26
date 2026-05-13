import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

def get_mask_index_wcy(masks):
    # masks[masks>0.5] = 1
    # masks[masks<=0.5] = 0
    # masked Region: 1
    vae_stride = (4, 8, 8)
    result_masks = []
    smaller_mask = []
    for mask in masks:
        c, depth, height, width = mask.shape
        new_depth = int((depth + 3) // vae_stride[0])
        height = 2 * (int(height) // (vae_stride[1] * 2))
        width = 2 * (int(width) // (vae_stride[2] * 2))

        # reshape
        mask = mask[0, :, :, :]
        mask = mask.view(depth, height, vae_stride[1], width, vae_stride[2])  # depth, height, 8, width, 8
        mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
        mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height, width)  # 8*8, depth, height, width

        mask = 1 - mask
        if depth%vae_stride[0] != 0:
            mask_res = mask[:, :depth%vae_stride[0]] # depth-depth%vae_stride[0]:]
            mask_res = mask_res.reshape(-1, 1, height, width)
            mask_res = torch.cumprod(mask_res, dim=0)[-1:]

            mask = mask[:, depth%vae_stride[0]:]
            mask = mask.reshape(vae_stride[1] * vae_stride[2], depth//vae_stride[0], vae_stride[0], height, width)
            mask = mask.permute(0, 2, 1, 3, 4)
            mask = mask.reshape(vae_stride[1]*vae_stride[2]*vae_stride[0], depth//vae_stride[0], height, width)
            mask = torch.cumprod(mask, dim=0)[-1:]
            mask = torch.cat([mask_res, mask], dim=1)

            result_masks.append(mask) # [1, 5, 60, 104]

        else:
            mask = mask.reshape(vae_stride[1] * vae_stride[2], depth//vae_stride[0], vae_stride[0], height, width)
            mask = mask.permute(0, 2, 1, 3, 4)
            mask = mask.reshape(vae_stride[1]*vae_stride[2]*vae_stride[0], depth//vae_stride[0], height, width)
            mask = torch.cumprod(mask, dim=0)
            result_masks.append(mask[-1:]) # [1, 5, 60, 104]

        wc, wd, wh, ww = result_masks[-1].shape
        this_mask = result_masks[-1].reshape([wc, wd, wh//2, 2, ww//2, 2]).permute(0, 1, 2, 4, 3, 5)
        this_mask = this_mask.reshape([wc, wd, wh//2, ww//2, 4])
        this_mask = torch.cumprod(this_mask, dim=-1)[:, :, :, :, -1]
        smaller_mask.append(this_mask)
    # return 1-torch.stack(result_masks, dim=0), 1-torch.stack(smaller_mask, dim=0) # [2, 1, 5, 480//8, 832//8]
    small_mask = 1-torch.stack(smaller_mask, dim=0)
    b, c, n, h, w = small_mask.shape
    big_mask = F.interpolate(small_mask.reshape(b*c*n, 1, h, w), size=(h*2, w*2), mode='nearest').reshape(b, c, n, 2*h, 2*w)
    return big_mask, small_mask

def enc_block_conc(masks):

    vae_stride = (4, 8, 8)
    masks = masks[:, :1]

    b, c, depth, height, width = masks.shape
    new_depth = int((depth + 3) // vae_stride[0])
    height = 2 * (int(height) // (vae_stride[1] * 2))
    width = 2 * (int(width) // (vae_stride[2] * 2))
    masks = 1-masks
    masks = masks.view(b, c, depth, height, vae_stride[1], width, vae_stride[2])
    masks = masks.permute(0, 1, 2, 3, 5, 4, 6)
    masks = masks.reshape(b, c, depth, height, width, vae_stride[1]*vae_stride[2])
    masks = torch.cumprod(masks, dim=5)[:, :, :, :, :, -1] # b, c, d, h, w
    masks = 1-masks

    masks = torch.cat([masks[:, :, :1], masks[:, :, :1], masks[:, :, :1], masks], dim=2) # [b, 1, d+3, h, w]
    return masks

def time_dev(vae_time_mask, strid):
    if strid > 0:
        vae_time_mask, _ = dilation_dev01(vae_time_mask, strid) # [b, 1, d+3, h, w]
    bb, cc, dd, hh, ww = vae_time_mask.shape
    vae_time_mask = vae_time_mask.repeat(1, 4, 1, 1, 1)
    vae_time_mask = vae_time_mask.reshape(bb, 4, dd//4, 4, hh, ww).permute(0, 3, 1, 2, 4, 5)
    # vae_time_mask = vae_time_mask.reshape(bb, 4, dd//4, 4, hh, ww).permute(0, 1, 3, 2, 4, 5)
    vae_time_mask = vae_time_mask.reshape(bb, 16, dd//4, hh, ww) # [b, 16, d, h, w]
    return vae_time_mask

def dilation_dev01(x, kernel_size = 3):
    # x: [b, c, d, h, w], 1 is the center
    b, c, d, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w)
    max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    dil_1 = max_pool(x)
    dil_0 = 1-max_pool(1-x)
    if kernel_size %2==0:
        dil_1 = dil_1[:, :, :h, :w]
        dil_0 = dil_0[:, :, :h, :w]
    return dil_1.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4), dil_0.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)

def processing_dila(mask, dila_st = 3):
    mask, _ = dilation_dev01(mask, dila_st)
    b, c, n, h, w = mask.shape
    mask = F.interpolate(mask.reshape(b*c*n, 1, h, w), size=(h//2, w//2), mode='nearest')
    mask = 1 - mask.reshape(b, c, n, h//2, w//2)
    # mask_tmp = mask.clone()

    # mask[:, :, :-1] = mask[:, :, :-1]*mask_tmp[:, :, 1: ]
    mask[:, :, 1: ] = mask[:, :, 1: ]*mask[:, :, :-1]
    mask = 1 - mask

    b, c, n, h, w = mask.shape
    big_mask = F.interpolate(mask.reshape(b*c*n, 1, h, w), size=(h*2, w*2), mode='nearest').reshape(b, c, n, h*2, w*2)
    return mask, big_mask

def encode_masks(masks):
    vae_stride = (4, 8, 8)
    result_masks = []
    for mask in masks:
        c, depth, height, width = mask.shape
        new_depth = int((depth + 3) // vae_stride[0])
        height = 2 * (int(height) // (vae_stride[1] * 2))
        width = 2 * (int(width) // (vae_stride[2] * 2))

        # reshape
        mask = mask[0, :, :, :]
        mask = mask.view(depth, height, vae_stride[1], width, vae_stride[1])  # depth, height, 8, width, 8
        mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
        mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height, width)  # 8*8, depth, height, width

        final_masks = []

        if depth % vae_stride[0] != 0:
            final_masks.append(mask[:, :1])

        # mask:torch.Size([64, 17, 60, 104]) npu:0
        zhengshu = int((depth//vae_stride[0])*vae_stride[0])
        this_mask = mask[:, 1:].reshape([vae_stride[1] * vae_stride[2], depth//vae_stride[0], vae_stride[0], height, width])
        this_mask = 1-torch.cumprod(1-this_mask, dim=2)[:, :, -1:].reshape([vae_stride[1] * vae_stride[2], depth//vae_stride[0], height, width])
        final_masks.append(this_mask)
        
        result_masks.append(torch.cat(final_masks, dim=1))
    return result_masks

def change4save(img):
    img = img.detach().cpu().numpy()
    img = img.squeeze()
    img = img.transpose([1,2,0])
    img = (img*255.0).astype('uint8')
    return img

def cv2_inpaint(videos, mask, test=False):
    # videos: [B, 3, F, H, W] -1 ~ 1, torch.tensor
    # mask  : [B, 1, F, H, W] 0 ~ 1, torch.tensor
    this_mask = mask.clone()
    # orign_video = videos.clone()

    b, c, f, h, w = videos.shape
    videos = (videos + 1)/2.
    videos = videos.permute(0, 2, 3, 4, 1).reshape([b*f, h, w, 3])
    videos = videos.cpu().numpy()

    mask = mask.permute(0, 2, 3, 4, 1).reshape([b*f, h, w])
    mask = mask.cpu().numpy()

    out = np.zeros([b*f, h, w, 3])

    for i in range(b*f):
        image = (videos[i]*255).astype('uint8')
        ori_mask = (mask[i]*255).astype('uint8')

        out_ = cv2.inpaint(image, ori_mask, 3, cv2.INPAINT_TELEA)
        # edge_mask = cv2.Canny(image, 100, 200)
        # edge_mask = cv2.dilate(edge_mask, np.ones((3,3), np.uint8))
        out[i] = out_#cv2.inpaint(out_, ori_mask, 3, cv2.INPAINT_NS)

    videos = torch.tensor(out).to(this_mask)
    videos = 2*videos/255.-1
    videos = videos.reshape([b, f, h, w, 3]).permute(0, 4, 1, 2, 3)
    return videos#*this_mask + (1-this_mask)*orign_video

def cv2_inpaint_inf(videos, mask, ds_mask):
    # videos: [B, 3, F, H, W] -1 ~ 1, torch.tensor
    # mask  : [B, 1, F, H, W] 0 ~ 1, torch.tensor
    # ds_mask: [B, 1, 5, H//8, W//8]

    # b, c, ff, hh, ww = ds_mask.shape
    # ds_mask = F.interpolate(ds_mask.reshape(b, c, ff, hh, ww), size=(ff*4, hh*16, ww*16), mode='nearest').reshape(b, c, ff*4, hh*16, ww*16)
    # ds_mask = ds_mask[:, :, 3:]
    b, c, f, h, w = mask.shape
    mask = mask.view(b, c, f, h//8, 8, w//8, 8)  # depth, height, 8, width, 8
    mask = mask.permute(0, 1, 2, 3, 5, 4, 6)  # 8, 8, depth, height, width
    mask = mask.reshape(b, c, f, h//8, w//8, 64)  # 8*8, depth, height, width

    mask = 1 - mask
    mask = torch.cumprod(mask, dim=-1)[:, :, :, :, :, -1]
    mask = F.interpolate(mask, size=(f, h, w), mode='nearest')
    mask = mask.repeat(1, 3, 1, 1, 1)

    # noise = torch.rand(videos.shape).to(videos)*2-1.
    noise = torch.mean(videos, dim=[3,4], keepdim=True)

    return videos*mask + (1-mask)*noise, 1-mask

def cv2_inpaint_infV2(videos, mask, ds_mask):
    # videos: [B, 3, F, H, W] -1 ~ 1, torch.tensor
    # mask  : [B, 1, F, H, W] 0 ~ 1, torch.tensor
    # ds_mask: [B, 1, 5, H//8, W//8]

    b, c, ff, hh, ww = ds_mask.shape
    ds_mask = F.interpolate(ds_mask.reshape(b, c, ff, hh, ww), size=(ff*4, hh*16, ww*16), mode='nearest').reshape(b, c, ff*4, hh*16, ww*16)
    ds_mask = ds_mask[:, :, 3:]
    ds_mask = ds_mask.repeat(1, 3, 1, 1, 1)

    noise = torch.rand(videos.shape).to(videos)*2-1.

    return videos*ds_mask + (1-ds_mask)*noise, 1-ds_mask

def cv2_inpaint_infV1(videos, mask, ds_mask):
    # videos: [B, 3, F, H, W] -1 ~ 1, torch.tensor
    # mask  : [B, 1, F, H, W] 0 ~ 1, torch.tensor
    # ds_mask: [B, 1, 5, H//8, W//8]

    b, c, ff, hh, ww = ds_mask.shape
    ds_mask = F.interpolate(ds_mask.reshape(b, c, ff, hh, ww), size=(ff*4, hh*16, ww*16), mode='nearest').reshape(b, c, ff*4, hh*16, ww*16)

    ds_mask = ds_mask[:, :, 3:]
    this_mask = mask.clone()
    # orign_video = videos.clone()

    b, c, f, h, w = videos.shape
    videos = (videos + 1)/2.
    videos = videos.permute(0, 2, 3, 4, 1).reshape([b*f, h, w, 3])
    videos = videos.cpu().numpy()

    mask = mask.permute(0, 2, 3, 4, 1).reshape([b*f, h, w])
    mask = mask.cpu().numpy()

    out = np.zeros([b*f, h, w, 3])

    for i in range(b*f):
        image = (videos[i]*255).astype('uint8')
        ori_mask = (mask[i]*255).astype('uint8')
        out[i] = cv2.inpaint(image, ori_mask, 3, cv2.INPAINT_TELEA)

    videos = torch.tensor(out).to(this_mask)
    videos = 2*videos/255.-1
    videos = videos.reshape([b, f, h, w, 3]).permute(0, 4, 1, 2, 3)
    noise = torch.rand(videos.shape).to(videos)*2-1.

    return videos*ds_mask + (1-ds_mask)*noise

def cv2_inpaint_v1(videos, mask, test=False):
    # videos: [B, 3, F, H, W] -1 ~ 1, torch.tensor
    # mask  : [B, 1, F, H, W] 0 ~ 1, torch.tensor
    this_mask = mask.clone()

    b, c, f, h, w = videos.shape
    videos = (videos + 1)/2.
    videos = videos.permute(0, 2, 3, 4, 1).reshape([b*f, h, w, 3])
    videos = videos.cpu().numpy()

    mask = mask.permute(0, 2, 3, 4, 1).reshape([b*f, h, w])
    mask = mask.cpu().numpy()

    out = np.zeros([b*f, h, w, 3])

    for i in range(b*f):
        if test:
            out[i] = cv2.inpaint((videos[i]*255).astype('uint8'), (mask[i]*255).astype('uint8'), 3, cv2.INPAINT_NS)#INPAINT_TELEA)
        else:
            out[i] = cv2.inpaint((videos[i]*255).astype('uint8'), (mask[i]*255).astype('uint8'), 3, cv2.INPAINT_TELEA)

    videos = torch.tensor(out).to(this_mask)
    videos = 2*videos/255.-1
    videos = videos.reshape([b, f, h, w, 3]).permute(0, 4, 1, 2, 3)
    return videos

def get_index_grad(index):
    # index: [num]
    num = index.shape[0]

    tmp = torch.linspace(-1.0, 1.0, num).to(index).type_as(index)
    short_index = tmp[index>0.5] # short_index: [ short_num ]
    short_num = short_index.shape[0]

    tmp_short = torch.linspace(-1.0, 1.0, short_num).to(index).type_as(index)
    tmp[index>0.5] = tmp_short
    return short_index, tmp

def index1d(x, index):
    # x    : [1, num, c]
    # index: [other_num]
    other_num = index.shape[0]
    xx = x.permute(0, 2, 1).unsqueeze(3) # [1, c, num, 1]
    index_ = index.reshape([1, other_num, 1, 1]) # [1, other_num, 1, 1]
    index_ = torch.cat([index_*0, index_], dim=-1).type_as(x)
    out = F.grid_sample(input=xx.float(), grid=index_.float(), mode='nearest', padding_mode='border')
    return out.squeeze(3).permute(0, 2, 1).type_as(x)

def get_index_grad_batch(index):
    # index: [b, num]
    b, num = index.shape[0], index.shape[1]
    delta_tmp = 1/(2*num)

    tmp = torch.linspace(delta_tmp-1.0, 1.0-delta_tmp, num).to(index.device).float()

    final_ = []
    final_len = []
    backward_ = []

    for i in range(b):
        short_index = tmp[index[i]>0.5] # short_index: [ short_num ]
        short_num = short_index.shape[0]

        final_.append(short_index)
        final_len.append(short_num)

    max_len = max(final_len)
    delta_max = 1/(2*max_len)
    for i in range(b):
        tmp_short = torch.linspace(delta_max-1.0, ((2-2*delta_max)/(max_len-1))*(final_len[i]-1)-1+delta_max, final_len[i]).to(index.device).float()
        this_tmp = tmp.clone()
        this_tmp[index[i]>0.5] = tmp_short
        backward_.append(this_tmp)

        if max_len != final_len[i]:
            add_tensor = torch.ones([max_len-final_len[i]]).to(final_[i].device)
            final_[i] = torch.cat([final_[i], add_tensor], dim=0)

    forward_index = torch.stack(final_, dim=0)          # [b, short_num]
    backward_index = torch.stack(backward_, dim=0)      # [b, orgin_num]
    return forward_index, backward_index 

def index1d_batch(x, index):
    # x    : [b, num, c]
    # index: [b, other_num]
    b, other_num = index.shape
    xx = x.permute(0, 2, 1).unsqueeze(3) # [b, c, num, 1]
    index_ = index.reshape([b, other_num, 1, 1]) # [b, other_num, 1, 1]
    index_ = torch.cat([index_*0, index_], dim=-1).to(x.device)
    out = F.grid_sample(input=xx.float(), grid=index_.float(), mode='nearest', padding_mode='reflection')
    return out.squeeze(3).permute(0, 2, 1).type_as(x)