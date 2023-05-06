import os
import torch
from omegaconf import OmegaConf

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.saving_utils import npz_to_video_grid
from scripts.sample_text2video import sample_text2video
from scripts.sample_utils import load_model
from lvdm.models.modules.lora import change_lora_v2


def save_results(videos, save_dir, 
                 save_name="results", save_fps=8
                 ):
    save_subdir = os.path.join(save_dir, "videos")
    os.makedirs(save_subdir, exist_ok=True)
    for i in range(videos.shape[0]):
        npz_to_video_grid(videos[i:i+1,...], 
                            os.path.join(save_subdir, f"{save_name}_{i:03d}.mp4"), 
                            fps=save_fps)
    video_path_list = [os.path.join(save_subdir, f"{save_name}_{i:03d}.mp4") for i in range(videos.shape[0])]
    print(f'Successfully saved videos in {video_path_list[0]}')
    return video_path_list
    

class Text2Video():
    def __init__(self,result_dir='./tmp/') -> None:
        self.download_model()
        config_file = 'models/base_t2v/model_config.yaml'
        ckpt_path = 'models/base_t2v/model_rm_wtm.ckpt'
        if os.path.exists('/dev/shm/model_rm_wtm.ckpt'):
            ckpt_path='/dev/shm/model_rm_wtm.ckpt'
        config = OmegaConf.load(config_file)
        self.lora_path_list = ['','models/videolora/lora_001_Loving_Vincent_style.ckpt',
                                'models/videolora/lora_002_frozenmovie_style.ckpt',
                                'models/videolora/lora_003_MakotoShinkaiYourName_style.ckpt',
                                'models/videolora/lora_004_coco_style_v2.ckpt']
        self.lora_trigger_word_list = ['','Loving Vincent style', 'frozenmovie style', 'MakotoShinkaiYourName style', 'coco style']
        model, _, _ = load_model(config, ckpt_path, gpu_id=0, inject_lora=False)
        self.model = model
        self.last_time_lora = ''
        self.last_time_lora_scale = 1.0
        self.result_dir = result_dir
        self.save_fps = 8
        self.ddim_sampler = DDIMSampler(model) 
        self.origin_weight = None

    def get_prompt(self, input_text, steps=50, model_index=0, eta=1.0, cfg_scale=15.0, lora_scale=1.0):
        torch.cuda.empty_cache()
        if steps > 60:
            steps = 60 
        if model_index > 0:
            input_text = input_text + ', ' + self.lora_trigger_word_list[model_index]
        inject_lora = model_index > 0
        self.origin_weight = change_lora_v2(self.model, inject_lora=inject_lora, lora_scale=lora_scale, lora_path=self.lora_path_list[model_index],
                    last_time_lora=self.last_time_lora, last_time_lora_scale=self.last_time_lora_scale, origin_weight=self.origin_weight)

        all_videos = sample_text2video(self.model, input_text, n_samples=1, batch_size=1,
                        sample_type='ddim', sampler=self.ddim_sampler,
                        ddim_steps=steps, eta=eta, 
                        cfg_scale=cfg_scale,
                        )
        prompt = input_text
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
        self.last_time_lora=self.lora_path_list[model_index]
        self.last_time_lora_scale = lora_scale
        video_path_list = save_results(all_videos, self.result_dir, save_name=prompt_str, save_fps=self.save_fps)
        return video_path_list[0]
    

