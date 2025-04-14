import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext(device=self.opts.device)
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="True"
import cv2
from utils import utils


class ShadowMapping(nn.Module):
    def __init__(self,opts,resolution):
        super(ShadowMapping, self).__init__()
        self.opts = opts
        self.resolution = resolution
        self.fov = 2*np.arctan(0.5*36/self.opts.focal_length)
        self.cot = np.tan(0.5*(np.pi - self.fov))

        self.rot_pi_origin_mat = utils.np2torch(np.array([[ np.cos(np.pi), 0, np.sin(np.pi), 0],
                                                    [ 0, 1, 0, 0],
                                                    [-np.sin(np.pi), 0,np.cos(np.pi), 0],
                                                    [ 0, 0, 0, 1]]),self.opts.device).to(torch.float32)
        self.m_dist_mat = utils.np2torch(self.translate(0,0,-self.opts.camera_distance),self.opts.device).to(torch.float32)
        self.local_mat = torch.matmul(self.rot_pi_origin_mat,self.m_dist_mat)
        m1_to_p1 = 2 * (torch.arange(self.resolution)/self.resolution + 1/(2*self.resolution)) - 1
        self.pos_map_x_tensor = m1_to_p1[None,None,None].repeat(1,1,self.resolution,1).to(self.opts.device) 
        self.pos_map_y_tensor = m1_to_p1[None,None,:,None].repeat(1,1,1,self.resolution).to(self.opts.device)
        self.ones = utils.np2torch(np.ones((1,1,self.resolution,self.resolution)),self.opts.device)
        self.idx_n_light = torch.arange(self.opts.n_light)[:,None,None].to(self.opts.device)
        M = 4
        self.N = 2*M
        self.idx_N = torch.arange(self.N)[None,:,None].to(self.opts.device)
        self.ck = np.pi * (2*(torch.arange(0, M)+1)-1)[None,:,None].to(self.opts.device)
        self.ones_scalar = torch.ones(1).to(self.opts.device)
        
        self.near = self.opts.camera_distance - np.sqrt(2) * self.opts.camera_distance * np.tan(self.fov/2)
        self.far = self.opts.camera_distance + np.sqrt(2) * self.opts.camera_distance * np.tan(self.fov/2)

        self.orth_proj = utils.np2torch(np.array([[1./self.opts.window, 0, 0, 0],
                                                    [0, 1./self.opts.window, 0, 0],
                                                    [0,  0, -2./(self.far-self.near), -(self.far+self.near)/(self.far-self.near)],
                                                    [0,  0,  0,  1]]),self.opts.device).to(torch.float32)
        self.offset = -0.0642233295781
        self.clamp_value = 0.4458709375254


    def forward(self,depth,als,mask,train=True):
        self.train = train
        if depth.shape[0] != als.shape[0] or als.shape[0] != mask.shape[0]:
            print('depth, als, mask shape mismatch')
            exit()
        batch_size = depth.shape[0]
        soft_shadow_maps_batch = torch.zeros((batch_size,self.opts.n_light,self.resolution,self.resolution)).to(self.opts.device)
        hard_shadow_maps_batch = torch.zeros((batch_size,self.opts.n_light,self.resolution,self.resolution)).to(self.opts.device)
        for b in range(batch_size):
            hard_shadow_maps_batch[b:b+1],soft_shadow_maps_batch[b:b+1] = self.make_light_area(depth[b:b+1],mask[b:b+1],als[b])
        soft_shadow_maps_batch = mask * soft_shadow_maps_batch
        return mask * hard_shadow_maps_batch, mask * soft_shadow_maps_batch


    def make_light_area(self,depth,mask,als):
        '''
        depth shape: [1,1,res,res]
        mask shape: [1,1,res,res]
        als shape: [n_light,7] (direction(3), sigma(1), intensity(3))
        output shape: [1,n_light,res,res]
        '''


        # Convert depth to mesh
        posw, idx = self.depth2mesh(depth,mask)


        # Convert the mesh to the light source view
        posw_center = torch.matmul(posw,self.local_mat.t()) 
        mv_mat_shadow_light = torch.einsum('xy,dyz->dxz',self.m_dist_mat,self.rotate_xyz_light(als[:,0:3]))
        posw_from_light = torch.einsum('vw,dxw->dvx',posw_center, mv_mat_shadow_light)
        posw_from_light = torch.einsum('dvw,xw->dvx',posw_from_light,self.self.orth_proj)


        # z_map: Depth map from light source
        _z_map, _ = dr.rasterize(glctx, posw_from_light.contiguous(), idx, resolution=[self.resolution, self.resolution])
        z_map = 0.5 * (1+_z_map[...,2])
        mask_fromlight = (z_map!=0.5).to(torch.float32)
        z_map[mask_fromlight==0] = 1
        # for uv to vertex
        u = (self.resolution * ((posw_from_light[:,:,0]+1)/2)).to(torch.long).clamp(0,self.resolution-1) # batch,i_idx
        v = (self.resolution * ((posw_from_light[:,:,1]+1)/2)).to(torch.long).clamp(0,self.resolution-1)# batch,i_idx
        

        
        # d: Distance from a point to the light source
        # z: Stored depth value in the shadow map
        # f: Binary shadow function
        d = (0.5 * (1 + posw_from_light[...,2]/posw_from_light[...,3])).clamp(0,1)
        z = z_map[self.idx_n_light[:,:,0], v, u] #[n_light,n_vtx]
        f = (d-z).clamp(0,None)
        ind_shadow = f > self.opts.shadow_threshold
        f[ind_shadow] = 0
        f[~ind_shadow] = 1


        
        # Approximate the binary function f with Fourier series expansion
        # a: Fourier coefficients
        # B_map: Fourier basis functions
        a = torch.zeros((self.opts.n_light,self.N,posw.shape[0])).to(self.opts.device)
        B_map = torch.zeros((self.opts.n_light,self.N,self.resolution,self.resolution)).to(self.opts.device)
        a[:,::2,:] = -2 * torch.sin(self.ck * (d[:,None,:] + self.offset)) / self.ck
        a[:,1::2,:] = 2 * torch.cos(self.ck * (d[:,None,:] + self.offset)) / self.ck
        B_map[:,::2] = torch.cos(self.ck[...,None] * z_map[:,None])
        B_map[:,1::2] = torch.sin(self.ck[...,None] * z_map[:,None])
        

        # Gaussian blur the Fourier basis functions
        sig = ((self.resolution/self.opts.resolution_optimize) * (6 * als[:,3] + 1) - 1) / 6
        D_map = self.fast_gaussian_blur(B_map, sig)

        
        # Compute the soft shadow function sf
        D = D_map[self.idx_n_light, self.idx_N, v[:,None,:], u[:,None,:]]
        q = torch.sum(a * D, dim=1).clamp(-self.clamp_value,self.clamp_value) / self.clamp_value
        sf = 0.5 * (q + 1)


        
        posw_zm = torch.matmul(posw,self.rot_pi_origin_mat.t())
        near = -posw_zm[:,2].max().item()
        far = -posw_zm[:,2].min().item()
        pers_proj = utils.np2torch(np.array([[self.cot, 0, 0, 0],
                                        [  0, self.cot, 0, 0],
                                        [  0,    0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
                                        [  0,    0,           -1,              0]]),self.opts.device).to(torch.float32)
        posw_fromcamera = torch.matmul(posw_zm,pers_proj.t())[None]
        depth_fromcamera = dr.rasterize(glctx, posw_fromcamera, idx, resolution=[self.resolution, self.resolution])[0]



        # f and sf are pasted on the mesh viewed from the camera
        f_map, _ = dr.interpolate(f[:,:,None], depth_fromcamera.expand(self.opts.n_light,-1,-1,-1), idx)
        sf_map, _ = dr.interpolate(sf[:,:,None], depth_fromcamera.expand(self.opts.n_light,-1,-1,-1), idx)
        if self.train:
            sf_map = dr.antialias(sf_map,
                                  depth_fromcamera.expand(self.opts.n_light,-1,-1,-1),
                                  posw_fromcamera.expand(self.opts.n_light,-1,-1),idx) #[b,h,w,4]
        f_map = f_map.clamp(0,1)
        sf_map = sf_map.clamp(0,1)

        return f_map[None,...,0], sf_map[None,...,0]

    def translate(self,x, y, z):
        return np.array([[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])

    def rotate_xyz_light(self,directions):
        x = directions[:,0]
        y = directions[:,1]
        z = directions[:,2]
        cosp = torch.sqrt(x*x+z*z)
        cost = (z/cosp)
        sint = (x/cosp)
    
        Rn = torch.zeros((self.opts.n_light,4,4)).to(self.opts.device)
        Rn[:,0,0] = cost
        Rn[:,0,1] = -sint*y
        Rn[:,0,2] = cosp*sint
        Rn[:,1,1] = cosp
        Rn[:,1,2] = y
        Rn[:,2,0] = -sint
        Rn[:,2,1] = -cost*y
        Rn[:,2,2] = cosp*cost
        Rn[:,3,3] = 1
        return Rn.permute(0,2,1)

    def depth2mesh(self,depth,mask):
        pos_map = torch.cat([- depth * np.tan(self.fov/2) * self.pos_map_x_tensor,
                             depth * np.tan(self.fov/2) * self.pos_map_y_tensor,
                             depth],dim=1)
        posw, idx = self.generate_mesh_indices_mask(pos_map, mask)
        return posw,idx

    def generate_mesh_indices_mask(self,pos_map, mask):
        mask_indices = torch.where(mask)
        n_masked_pixels = int(torch.sum(mask).item())
        idx_map = -self.ones
        idx_map[mask_indices] = torch.arange(0, n_masked_pixels).to(self.opts.device,torch.float32)

        idx_map00 = idx_map[:,:,:-1,:-1]
        idx_map01 = idx_map[:,:,:-1,1:]
        idx_map11 = idx_map[:,:,1:,1:]
        idx_map10 = idx_map[:,:,1:,:-1]

        connected_00_01 = torch.abs(pos_map[:,2:3,:-1,:-1] - pos_map[:,2:3,:-1,1:]) < self.opts.depth_threshold 
        connected_01_11 = torch.abs(pos_map[:,2:3,:-1,1:] - pos_map[:,2:3,1:,1:]) < self.opts.depth_threshold 
        connected_00_11 = torch.abs(pos_map[:,2:3,:-1,:-1] - pos_map[:,2:3,1:,1:]) < self.opts.depth_threshold
        connected_10_11 = torch.abs(pos_map[:,2:3,1:,:-1] - pos_map[:,2:3,1:,1:]) < self.opts.depth_threshold 
        connected_00_10 = torch.abs(pos_map[:,2:3,:-1,:-1] - pos_map[:,2:3,1:,:-1]) < self.opts.depth_threshold 

        lower_tri_idx_map = torch.cat((idx_map00,idx_map11,idx_map01), 1)
        upper_tri_idx_map = torch.cat((idx_map00,idx_map10,idx_map11), 1)
        lower_tri_mask = ((idx_map00>=0) * (idx_map01>=0) * (idx_map11>=0) * connected_00_01 * connected_01_11 * connected_00_11)
        upper_tri_mask = ((idx_map00>=0) * (idx_map11>=0) * (idx_map10>=0) * connected_00_11 * connected_10_11 * connected_00_10)
        lower_tri_indices = torch.stack([lower_tri_idx_map[:,0:1][lower_tri_mask],
                                       lower_tri_idx_map[:,1:2][lower_tri_mask],
                                       lower_tri_idx_map[:,2:3][lower_tri_mask]],1)
        upper_tri_indices = torch.stack([upper_tri_idx_map[:,0:1][upper_tri_mask],
                                       upper_tri_idx_map[:,1:2][upper_tri_mask],
                                       upper_tri_idx_map[:,2:3][upper_tri_mask]],1)
        idx = torch.cat([lower_tri_indices, upper_tri_indices], 0).to(torch.int32)


        pos = torch.stack([pos_map[:,0:1][mask_indices],
                        pos_map[:,1:2][mask_indices],
                        pos_map[:,2:3][mask_indices]], 1)

        posw = torch.cat([pos,torch.ones((n_masked_pixels,1)).to(self.opts.device)],1)
        return posw, idx

    # for debug
    def save_ply(self,save_path, vertices, faces=None):
        '''
        vertices: [n_verts, 3]
        faces: [n_faces, 3]
        '''

        if len(vertices.shape) != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
            print('Error: vertices.shape should be (#verts, 3)')
            return
        
        n_verts = vertices.shape[0]
        
        if faces is not None:
            if len(faces.shape) != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
                print('Error: faces.shape should be (#faces, 3)')
                return

        # recording data as "Structured Arrays"
        # see https://numpy.org/doc/stable/user/basics.rec.html
        with open(save_path, 'wb') as ply:
            header  = [b'ply']
            header += [b'format binary_little_endian 1.0']
            header += [b'comment Generated using simple script written by yknmr']

            header += [b'element vertex %d' % n_verts]
            header += [b'property float %s' % p for p in [b'x',b'y',b'z']]
            
            out_dtype = '3float32'
            out_list = [vertices]

            if faces is not None:
                n_faces = faces.shape[0]
            
                header += [b'element face %d' % n_faces]
                header += [b'property list uchar int vertex_index']
                
                out_face = np.empty(n_faces, dtype='uint8, 3int32')
                out_face['f0'] = 3 * np.ones(n_faces)
                out_face['f1'] = faces
            
            header += [b'end_header\n']
            
            ply.write(b'\n'.join(header))

            out_data = np.empty(n_verts, dtype=out_dtype)
            if len(out_list) == 1:
                out_data = out_list[0]
            else:
                for idx, data in enumerate(out_list):
                    out_data['f%d' % idx] = data
            
            ply.write(out_data.tostring())
            if faces is not None:
                ply.write(out_face.tostring())


    def make_gaussian_kernel(self, kernel_size, sigma):
        '''
        kernel_size shape: [1]
        sigma shape: [1]
        '''

        ts = torch.linspace(-kernel_size // 2, kernel_size // 2 + 1, kernel_size).to(self.opts.device) #ここのサイズが可変
        gauss = torch.exp(-(ts / sigma)**2 / 2)
        if kernel_size == 1:
            kernel = self.ones_scalar
        else:
            kernel = gauss / gauss.sum()
        return kernel


    def fast_gaussian_blur(self, img, sigma):
        '''
        img shape: [n_light,n_channel,res,res]
        sigma shape: [n_light]
        output shape: [n_light,n_channel,res,res]
        '''

        kernel_size = 2 * torch.ceil(3 * sigma).to(torch.int32) + 1
        list_img = []
        groups = img.shape[1]
        for i, (ks, s) in enumerate(zip(kernel_size, sigma)):
            kernel = self.make_gaussian_kernel(ks, s)
            pad = ks // 2
            im = F.pad(img[i:i+1], [pad, pad, pad, pad], mode="reflect")
            im = F.conv2d(im, torch.tile(kernel.view(1, 1, ks, 1),(groups,1,1,1)),groups=groups)
            im = F.conv2d(im, torch.tile(kernel.view(1, 1, 1, ks),(groups,1,1,1)),groups=groups)
            list_img.append(im)
        blurred = torch.cat(list_img,dim=0)
        return blurred # [d,N,res,res] 