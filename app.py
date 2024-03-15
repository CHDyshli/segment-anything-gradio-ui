import gradio as gr
import gradio_image_prompter as gr_ext
# import os
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch



title = "a SAM gradio demo"
header = (
    "<div align='center'>"
    "<h1>SAM gradio app(a demo)</h1>"
    "</div>"
)
theme = "soft"
css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
            #anno-img .mask.active {opacity: 0.7}"""



def get_added_image(masks:list, image:np.ndarray):
    if len(masks)==0:
        return image
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    mask_all = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.random.random(3).tolist()
        mask_all[m] =  color_mask
    added_img = image /255* 0.5 + mask_all*0.5
    return added_img

def on_auto_submit_btn(auto_input_img):
    sam_checkpoint = "./models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(auto_input_img)
    added_img = get_added_image(masks, auto_input_img)
    return added_img



def on_click_submit_btn(click_input_img):

    # set sam
    sam_checkpoint = "./models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(click_input_img['image'])

    # set points
    np_points = np.array(click_input_img['points'])
    positive_point_conditon = (np_points[:, 2]==1) & (np_points[:, 5]==4)
    positive_points = np_points[positive_point_conditon]
    positive_points = positive_points[:, :2].tolist()
    negative_point_conditon = (np_points[:, 2]==0) & (np_points[:, 5]==4)
    negative_points = np_points[negative_point_conditon]
    negative_points = negative_points[:, :2].tolist()
    box_condition = (np_points[:, 2]==2) & (np_points[:, 5]==3)
    box_points = np_points[box_condition]
    box_points = box_points[:, [0, 1, 3, 4]].tolist()
    input_boxes = torch.tensor(box_points).to(device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, click_input_img['image'].shape[:2])
    masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
    )
    mask = masks[0].cpu().numpy().squeeze().astype(np.uint8)*255


    return mask

def on_auto_test_btn(auto_input_img):
    return auto_input_img.shape

def on_click_reset_btn():
    return None, None


examples = [["examples/chang'an univ.png"], ["examples/chd_weishui1.jpg"], ["examples/chd_weishui2.jpg"]]
click_examples = [{"image":"examples/chang'an univ.png"}, 
                  {"image":"examples/chd_weishui1.jpg"}, 
                  {"image":"examples/chd_weishui2.jpg"}] 

with gr.Blocks(title=title, theme=theme, css=css) as demo:
    gr.Markdown(header)

    with gr.Row():
        with gr.Column():

            with gr.Tab(label="Automatic") as auto_tab:
                with gr.Row():
                    auto_input_img = gr.Image(label="Input Image", 
                        sources='upload',
                        height=400, 
                        width=500,
                        show_label=True
                        )
                    # auto_output_anno_img = gr.AnnotatedImage(label="Output Image")
                    auto_output_img = gr.Image(
                        label="Output Image", 
                        interactive=False, 
                        height=400, 
                        width=500, 
                        show_label=True,
                        show_download_button=True
                        )
                with gr.Row():
                    auto_clr_btn=gr.ClearButton(components=[auto_input_img, auto_output_img])
                    auto_submit_btn = gr.Button("Submit")
               
                auto_submit_btn.click(
                    fn=on_auto_submit_btn,
                    inputs=[auto_input_img],
                    outputs=[auto_output_img]
                )

                with gr.Row():
                    gr.Examples(examples=examples,
                                inputs=[auto_input_img],
                                # outputs=[auto_output_img],
                                # fn=segment_everything,
                                # cache_examples=True,
                                examples_per_page=3
                                )

              

            with gr.Tab("Box") as click_tab:
                with gr.Row():
                    click_input_img = gr_ext.ImagePrompter(
                        show_label=True,
                        label="Input Image",
                        # height=400,
                        # width=500,
                        interactive=True,
                        sources='upload'
                    )
                    click_output_img = gr.Image(
                        show_label=True,
                        label="Output Image", 
                        interactive=False, 
                        # height=400, 
                        # width=500,
                        show_download_button=True
                        )
                with gr.Row():
                    click_clr_btn=gr.ClearButton(components=[click_input_img, click_output_img])
                    # click_reset_btn = gr.Button("Clear")
                    click_submit_btn = gr.Button("Submit")
                
                click_submit_btn.click(
                    fn=on_click_submit_btn,
                    inputs=[click_input_img],
                    outputs=[click_output_img]
                )
                with gr.Row():
                    gr.Examples(examples=click_examples,
                                inputs=[click_input_img],
                                # outputs=[auto_output_img],
                                # fn=segment_everything,
                                # cache_examples=True,
                                examples_per_page=3
                                )
    


                
                

        


if __name__ == "__main__":
    demo.launch()