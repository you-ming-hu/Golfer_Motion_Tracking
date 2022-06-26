import matplotlib.pyplot as plt
import pathlib
import numpy as np


def SegWithClass(save_root_path,batch_data,output):
    batch_size = batch_data['image'].shape[0]
    batch_data['mask'] = batch_data['mask'].cpu().numpy()
    batch_data['contrast_exist'] = batch_data['contrast_exist'].cpu().numpy()
    batch_data['image'] = batch_data['image'].cpu().numpy()
    
    for i in range(batch_size):
        mask = batch_data['mask'][i]
        contrast_exist = batch_data['contrast_exist'][i]
        curr_img = batch_data['image'][i][0]
        prev_img = batch_data['image'][i][1]
        
        contrast_exist = batch_data['contrast_exist'][i]
        sample_id = batch_data['sample_id'][i]
        frame_id = batch_data['frame_id'][i]
        
        pred_mask = output['mask'][i]
        pred_contrast_exist =  output['contrast_exist'][i]
        
        if contrast_exist == 1:
            need_save = bool(np.random.binomial(1,0.33))
        else:
            need_save = bool(np.random.binomial(1,0.1))
            
        if need_save:
            save_path = pathlib.Path(save_root_path,sample_id,frame_id)
            save_path.mkdir(parents=True,exist_ok=True)
            
            plt.imsave(save_path.joinpath(f'prev.png'),prev_img,cmap='gray')
            plt.imsave(save_path.joinpath(f'curr.png'),curr_img,cmap='gray')
            plt.imsave(save_path.joinpath(f'mask_{int(contrast_exist):d}.png'),mask,cmap='gray')
            
            plt.imsave(save_path.joinpath(f'pred_{pred_contrast_exist:.4f}.png'),pred_mask,cmap='gray')
        
        
        
        