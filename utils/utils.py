import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F


def np2torch(img,device='cuda'):
    if img is None:
        print("The loaded image is empty.")
        exit()
    img = torch.from_numpy(img).clone().to(device, non_blocking=True, dtype=torch.float)
    return img


def torch2np(img):
    img = img.to('cpu').detach().numpy().copy()
    return img


def lrgb2srgb(img):
    if isinstance(img, torch.Tensor):
        img_copy = img.clone()
        srgb = torch.pow(img_copy,1/2.2)
    else:
        img_copy = img.copy()
        srgb = pow(img_copy,1/2.2)
    return srgb


def lrgb2gray(img):
    if isinstance(img, torch.Tensor):
        gray = 0.2126*img[:,0,:,:] + 0.7152*img[:,1,:,:] + 0.0722*img[:,2,:,:]
    else:
        gray = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2] 
    return gray


def generate_log_txt(save_path,label_list=[]):
    open(save_path, 'w').close()
    with open(save_path,"a") as f:
        n_label = len(label_list) 
        if n_label != 0:
            for i in range(n_label):
                if i == n_label - 1:
                    f.write('%s' % label_list[i])
                else:
                    f.write('%s,' % label_list[i])
            f.write('\n')


def fibonacci_sphere(samples=1000):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)

        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)


def rotation_als(light,theta,vertical=False):
    light_rot = light.clone()
    if vertical:
        rot_mat = torch.tensor([[1,0,0],
                                [0,np.cos(theta),np.sin(theta)],
                                [0,-np.sin(theta),np.cos(theta)]]).to(light.device, dtype=torch.float)
    else:
        rot_mat = torch.tensor([[np.cos(theta),0,-np.sin(theta)],
                                [0,1,0],
                                [np.sin(theta),0,np.cos(theta)]]).to(light.device, dtype=torch.float)
    light_rot[:,:,0:3] = torch.einsum('yx,bdx->bdy',rot_mat,light[:,:,0:3])
    return light_rot

def img2xlsx(save_path,img_gray):
    '''
    Check the pixel values (for debugging)
    img_gray: numpy array
        shape: [h,w]
    '''
    import openpyxl
    wb = openpyxl.Workbook()
    sheet = wb.active
    print('writing to xlsx...')
    for y in range(img_gray.shape[0]):
        for x in range(img_gray.shape[1]):
            sheet.cell(row=y+1,column=x+1,value=img_gray[y,x])   
    print('saving %s...' % save_path)
    wb.save(save_path)
    print('finish')

def get_env_normal(h,w,device='cuda'):
    '''
    Generate a normal map corresponding to the environment map in latitude and longitude format
    h: height of the environment map
    w: width of the environment map
    
    output shape: [1,3,h,w]
    '''

    w_repeat = torch.tile(torch.arange(0,w,1)[None,None,None,:],(1,1,h,1)) + 1/2 #[0.5,...,w-0.5]を縦に伸ばす
    h_repeat = torch.tile(torch.arange(0,h,1)[None,None,:,None],(1,1,1,w)) + 1/2 #[0.5,...,h-0.5]を横に伸ばす

    phi = np.pi/2 * (2*h_repeat/h - 1) #[pi/2 * (1/h - 1), pi/2 * ((2h-1)/h - 1)] #[-90,+90]
    theta = np.pi * (2*w_repeat/w - 1) # [-180,180]

    normal_x = torch.cos(phi) * torch.cos(theta)
    normal_y = torch.sin(phi)
    normal_z = torch.cos(phi) * torch.sin(theta)
    normal_map = torch.cat([normal_x,normal_y,normal_z],dim=1)
    return normal_map.to(device)

def mapping_als(env, als, circle_size_min=0.001, circle_size_max=0.3, circle_edge_size=np.pi/128):
    '''
    env shape: [1,3,h,w]
    als shape: [1,n_light,7] (direction(3), sigma(1), intensity(3))
    circle_size_min: sigma が最も小さい箇所での 円のサイズ [0,2]
    circle_size_max: sigma が最も大きい箇所での 円のサイズ [0,2]
    circle_edge_size: 円のエッジのサイズ
    
    output: numpy array
        shape: [h,w,3]
        range: [0,255]
    '''

    intensity_gray = 0.2126*als[0,:,4] + 0.7152*als[0,:,5] + 0.0722*als[0,:,6] #[d]
    intensity_norm = 255 * (intensity_gray - torch.min(intensity_gray)) / (torch.max(intensity_gray) - torch.min(intensity_gray)) #[d,3],[0-255]
    
    normal_env = get_env_normal(env.shape[2], env.shape[3], device=als.device)[0]
    mapping_dls = torch.einsum('xhw,dx->dhw',normal_env,als[0,:,0:3]).clamp(0,1) #[d,h,w]

    sigma_norm = (als[0,:,3] - torch.min(als[0,:,3])) / (torch.max(als[0,:,3]) - torch.min(als[0,:,3])) #[0-1]
    cos_inner = (circle_size_min - circle_size_max) * sigma_norm[:,None,None].expand(-1,env.shape[2],env.shape[3]) + (1 - circle_size_min)

    mapping_dls_inner = mapping_dls > cos_inner #[d,h,w]
    cos_outer = torch.cos(torch.acos(cos_inner) + circle_edge_size)
    mapping_dls_outer = mapping_dls > cos_outer #[d,h,w]
    mapping_dls = torch2np(torch.logical_xor(mapping_dls_outer,mapping_dls_inner)) #[d,h,w]

    # Change the color according to the intensity of the light
    col_map = np.zeros((env.shape[2], env.shape[3], 3), dtype=np.float32)
    for inten_norm,map_dls in zip(intensity_norm,mapping_dls):
        color = cv2.applyColorMap(torch2np(inten_norm[None]).astype(np.uint8), colormap=cv2.COLORMAP_TURBO)[...,[2,1,0]]
        col_map[np.tile(map_dls[:,:,None],(1,1,3))] = (color * map_dls[:,:,None])[np.tile(map_dls[:,:,None],(1,1,3))]
    col_map[col_map==0] = torch2np(env[0].permute(1,2,0))[col_map==0]
    return col_map.clip(0,255)


def get_rect(mask): #mask: [1,1,h,w]
    a,b = torch.where(mask[0,0])
    y_min = torch.min(a)
    y_max = torch.max(a)
    x_min = torch.min(b)
    x_max = torch.max(b)
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    return y_min,x_min,h,w

def centering_img(img,mask,pad_h=10,new_res=1024):
    '''
    img shape: [1,n_channel,h,w]
    mask shape: [1,1,h,w]
    pad_h: padding height
    new_res: new resolution

    output shape: [1,n_channel,new_res,new_res], [1,1,new_res,new_res]
    '''

    _,n_c,res_h,res_w = img.shape
    _,_,_res_h,_res_w = mask.shape
    if res_h != _res_h or res_w != _res_w:
        print('mask size is not same as img size')
        print('img: ', img.shape[2:4])
        print('mask: ', mask.shape[2:4])
        exit()

    # get the center of the mask
    y_start, x_start, h, w = get_rect(mask)
    y_end = y_start + h

    if w > h:
        print('The rectangle should be vertical. Please check the mask.')
        exit()

    # get the size of the square to be trimmed
    _new_res = h
    if y_start > 0: #if there is a gap on the top
        _new_res += min(pad_h, y_start)
    if y_end < res_h: #if there is a gap on the bottom
        _new_res += min(pad_h, res_h - y_end)


    y_trim_start = max(0,y_start - pad_h)
    y_trim_end = y_trim_start+_new_res
    

    x_trim_start = x_start + (w - _new_res) // 2
    insert_x_start = max(0, - x_trim_start)
    x_trim_start = max(0,x_trim_start)

    x_trim_end = x_start + (w + _new_res) // 2
    insert_x_end = min(_new_res, _new_res - (x_trim_end - res_w))
    x_trim_end = min(x_trim_end,res_w)


    img_centered = torch.zeros((1, n_c, _new_res, _new_res), dtype=torch.float).to(img.device)
    mask_centered = torch.zeros((1, 1, _new_res, _new_res), dtype=torch.float).to(img.device)
    img_centered[:,:,:,insert_x_start:insert_x_end] = img[:,:,y_trim_start:y_trim_end,x_trim_start:x_trim_end]
    mask_centered[:,:,:,insert_x_start:insert_x_end] = mask[:,:,y_trim_start:y_trim_end,x_trim_start:x_trim_end]

    # resize
    img_centered = F.interpolate(img_centered, size=(new_res, new_res), mode='bilinear')
    mask_centered = F.interpolate(mask_centered, size=(new_res, new_res), mode='nearest')
    
    return img_centered, mask_centered


def clip_img(img,mask):
    '''
    img shape: [1,n_channel,h,w]
    mask shape: [1,1,h,w]
    output shape: [1,n_channel,h',w'], [1,1,h',w']
    '''
    _,b = torch.where(mask[0,0])
    min_ind = torch.min(b)
    max_ind = torch.max(b) + 1

    # make sure the size is even because of ffmpeg
    if (max_ind - min_ind) % 2 != 0:
        max_ind -= 1

    img_clipped = img[...,min_ind:max_ind]
    mask_clipped = mask[...,min_ind:max_ind]
    return img_clipped, mask_clipped

def generate_video(save_video_path, frame_paths, fps, disp_log=False):
    if disp_log:
        os.system('ffmpeg -y -framerate %f -i %s -c:v libx264 -pix_fmt yuv420p %s' % (fps, frame_paths, save_video_path))
    else:
        os.system('ffmpeg -y -framerate %f -i %s -c:v libx264 -pix_fmt yuv420p -loglevel fatal %s' % (fps, frame_paths, save_video_path))