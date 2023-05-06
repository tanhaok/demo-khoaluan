import os
import sys
import gradio as gr
from demo import Text2Video
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))

t2v_examples = [
    ['an elephant is walking under the sea, 4K, high definition',50,'origin',1,15,1,],
    ['an astronaut riding a horse in outer space',25,'origin',1,15,1,],
    ['a monkey is playing a piano',25,'vangogh',1,15,1,],
    ['A fire is burning on a candle',25,'frozen',1,15,1,],
    ['a horse is drinking in the river',25,'yourname',1,15,1,],
    ['Robot dancing in times square',25,'coco',1,15,1,],                    
]


def generate_video_demo(result_dir='./tmp/'):
    text2video = Text2Video(result_dir)
    with gr.Blocks(analytics_enabled=False) as gui:
        gr.Markdown("<div align='center'> <h2> Demo Khoa Luan Tot Nghiep - K19 - SPKT </h2> <br /> <h2>De Tai: Tao Video Tu Van Ban</h2> <br /> <a style='font-size:18px;color: #000000' href='https://github.com/tanhaok/demo-khoaluan'> Github </div>")
        
        with gr.Tab(label="Text to Video"):
            with gr.Column():
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        input_text = gr.Text(label='Prompts')
                        model_choices = ['origin', 'vangogh',
                                         'frozen', 'yourname', 'coco']
                        with gr.Row():
                            model_index = gr.Dropdown(
                                label='Models', elem_id=f"model", choices=model_choices, value=model_choices[0], type="index", interactive=True)
                        with gr.Row():
                            steps = gr.Slider(
                                minimum=1, maximum=60, step=1, elem_id=f"steps", label="Sampling steps", value=50)
                            eta = gr.Slider(
                                minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="eta")
                        with gr.Row():
                            lora_scale = gr.Slider(
                                minimum=0.0, maximum=2.0, step=0.1, label='Lora Scale', value=1.0, elem_id="lora_scale")
                            cfg_scale = gr.Slider(
                                minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=15.0, elem_id="cfg_scale")
                        send_btn = gr.Button("Send")
                    with gr.Tab(label='result'):
                        output_video_1 = gr.Video().style(width=384)
                gr.Examples(examples=t2v_examples,
                            inputs=[input_text, steps, model_index,
                                    eta, cfg_scale, lora_scale],
                            outputs=[output_video_1],
                            fn=text2video.get_prompt,
                            cache_examples=False)
                # cache_examples=os.getenv('SYSTEM') == 'spaces')
            send_btn.click(
                fn=text2video.get_prompt,
                inputs=[input_text, steps, model_index,
                        eta, cfg_scale, lora_scale,],
                outputs=[output_video_1],
            )
            
        ####### text to video personal #######
        with gr.Tab(label="Text to Video"):
            with gr.Column():
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        input_text = gr.Text(label='Prompts')
                        model_choices=['origin','vangogh','frozen','yourname', 'coco']
                        with gr.Row():
                            model_index = gr.Dropdown(label='Models', elem_id=f"model", choices=model_choices, value=model_choices[0], type="index",interactive=True)
                        with gr.Row():
                            steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id=f"steps", label="Sampling steps", value=50)
                            eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="eta")
                        with gr.Row():
                            lora_scale = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label='Lora Scale', value=1.0, elem_id="lora_scale")
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=15.0, elem_id="cfg_scale")
                        send_btn = gr.Button("Send")
                    with gr.Tab(label='result'):
                        output_video_1 =  gr.Video().style(width=384)
                gr.Examples(examples=t2v_examples,
                            inputs=[input_text,steps,model_index,eta,cfg_scale,lora_scale],
                            outputs=[output_video_1],
                            fn=text2video.get_prompt,
                            cache_examples=False)
                        #cache_examples=os.getenv('SYSTEM') == 'spaces')
            send_btn.click(
                fn=text2video.get_prompt, 
                inputs=[input_text,steps,model_index,eta,cfg_scale,lora_scale,],
                outputs=[output_video_1],
            )
        #######videocontrol######  change to tune-a-video
        # with gr.Tab(label='Tune-A_video'):
        #     with gr.Column():
        #         with gr.Row():
        #             # with gr.Tab(label='input'):
        #             with gr.Column():
        #                 with gr.Row():
        #                     vc_input_video = gr.Video(label="Input Video").style(width=256)
        #                     vc_origin_video = gr.Video(label='Center-cropped Video').style(width=256)
        #                 with gr.Row():
        #                     vc_input_text = gr.Text(label='Prompts')
        #                 with gr.Row():
        #                     vc_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="vc_eta")
        #                     vc_cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=15.0, elem_id="vc_cfg_scale")
        #                 with gr.Row():
        #                     vc_steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="vc_steps", label="Sampling steps", value=50)
        #                     frame_stride = gr.Slider(minimum=0 , maximum=100, step=1, label='Frame Stride', value=0, elem_id="vc_frame_stride")
        #                 with gr.Row():
        #                     resolution = gr.Slider(minimum=128 , maximum=512, step=8, label='Long Side Resolution', value=256, elem_id="vc_resolution")
        #                     video_frames = gr.Slider(minimum=8 , maximum=64, step=1, label='Video Frame Num', value=16, elem_id="vc_video_frames")
        #                 vc_end_btn = gr.Button("Send")
        #             with gr.Tab(label='Result'):
        #                 vc_output_info = gr.Text(label='Info')
        #                 with gr.Row():
        #                     vc_depth_video = gr.Video(label="Depth Video").style(width=256)
        #                     vc_output_video = gr.Video(label="Generated Video").style(width=256)

        #         gr.Examples(examples=control_examples,
        #                     inputs=[vc_input_video, vc_input_text, frame_stride, vc_steps, vc_cfg_scale, vc_eta, video_frames, resolution],
        #                     outputs=[vc_output_info, vc_origin_video, vc_depth_video, vc_output_video],
        #                     fn = videocontrol.get_video,
        #                     cache_examples=os.getenv('SYSTEM') == 'spaces',
        #         )
        #     vc_end_btn.click(inputs=[vc_input_video, vc_input_text, frame_stride, vc_steps, vc_cfg_scale, vc_eta, video_frames, resolution],
        #                     outputs=[vc_output_info, vc_origin_video, vc_depth_video, vc_output_video],
        #                     fn = videocontrol.get_video
        #     )

    return gui

if __name__ == "__main__":
    result_dir = os.path.join('./', 'results')
    gui = generate_video_demo(result_dir)
    gui.queue(concurrency_count=1, max_size=10)
    gui.launch()