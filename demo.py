import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import argparse


def anyres_preprocess_multi_scale(images, image_processor, max_pixels=-1, min_pixels=-1, scale_downsample_ratio=0.7071):
    assert min_pixels > 0 and max_pixels > 0, 'min_pixels and max_pixels must be set'
    if not isinstance(images, list):
        images = [images]
    
    pixel_values_all, image_grid_thws_all, num_scales_all = [], [], []
    for image in images:
        ret = image_processor(image, return_tensors="pt", min_pixels=min_pixels, max_pixels=max_pixels)
        image_grid_thws = [ret['image_grid_thw'][0]]
        pixel_values = ret['pixel_values'].reshape(ret['image_grid_thw'].prod(), -1, image_processor.patch_size, image_processor.patch_size)

        while True:
            current_pixels = image_grid_thws[0].prod() * (image_processor.patch_size ** 2)
            max_pixels = current_pixels * (scale_downsample_ratio ** 2)
            if max_pixels < min_pixels:
                break
            ret = image_processor(image, return_tensors="pt", min_pixels=min_pixels, max_pixels=max_pixels)
            if ret['image_grid_thw'].prod() >= image_grid_thws[0].prod():
                break
            image_grid_thws.insert(0, ret['image_grid_thw'][0])
            pixel_values = torch.cat([ret['pixel_values'].reshape(ret['image_grid_thw'].prod(), -1, image_processor.patch_size, image_processor.patch_size), pixel_values], dim=0)
            
        pixel_values_all.append(pixel_values)
        image_grid_thws_all.extend(image_grid_thws)
        num_scales_all.append(len(image_grid_thws))
    pixel_values = torch.cat(pixel_values_all, dim=0)
    return pixel_values, image_grid_thws_all, num_scales_all


def load_image(
    image_files,
    image_processor,
    patch_size=16,
    max_num=24576,
    min_num=256,
    upscale=False,
    scale_downsample_ratio=0.7071,
):
    
    if not isinstance(image_files, list):
        image_files = [image_files]
    
    images = []
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        if upscale:
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
        images.append(image)
    
    min_pixels = min_num * (patch_size ** 2)
    max_pixels = max_num * (patch_size ** 2)
    pixel_values, image_grid_thws, num_scales = anyres_preprocess_multi_scale(
                                                                images=images,
                                                                image_processor=image_processor,
                                                                max_pixels=max_pixels,
                                                                min_pixels=min_pixels,
                                                                scale_downsample_ratio=scale_downsample_ratio,
                                                            )

    image_grid_thws = torch.stack(image_grid_thws)
    num_scales = torch.tensor(num_scales)
    return pixel_values, image_grid_thws, num_scales


def load_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    device = torch.cuda.current_device()
    model = AutoModel.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=False
    ).eval()
    model.init_special_token_ids(tokenizer)

    # fix bug caused by size mismatch
    if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
        model.language_model.tie_weights()

    model = model.to(device)

    return model, tokenizer


def generate(message, model, tokenizer):
    image_num = len([x for x in message if x['type'] == 'image'])
    prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])

    if image_num > 0:
        image_paths = [x['value'] for x in message if x['type'] == 'image']
        pixel_values, image_grid_thws, num_scales = load_image(
                                                                image_paths,
                                                                model.image_processor,
                                                                max_num=model.config.max_dynamic_patch,
                                                                min_num=model.config.min_dynamic_patch,
                                                                patch_size=model.config.vision_config.patch_size,
                                                                scale_downsample_ratio=model.config.scale_downsample_ratio,
                                                            )
        pixel_values = pixel_values.cuda().to(torch.bfloat16)
        image_grid_thws = image_grid_thws.cuda()
        num_scales = num_scales.cuda()
    else:
        pixel_values, image_grid_thws, num_scales = None, None, None

    generation_config = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
    with torch.no_grad():
        try:
            response = model.chat(
                tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=generation_config,
                verbose=True,
                anyres_image_size=True,
                num_patches_list=image_grid_thws,
                num_scales=num_scales,
                )
        except Exception as e:
            print(f"Error in model chat: {e}")
            raise e
    return response


def main():
    parser = argparse.ArgumentParser(description="Generate a response from a multimodal model.")
    parser.add_argument("--model_name_or_path", type=str, default="OpenGVLab/NaViL-2B", help="Name or Path to the pretrained model.")
    args = parser.parse_args()

    model, tokenizer = load_model_tokenizer(args.model_name_or_path)

    message = [
        {"type": "image", "value": "./examples/image1.jpg"},
        {"type": "text", "value": "Please describe the image shortly."},
    ]

    response = generate(message, model, tokenizer)

    print("=== Response ===")
    print(response)


if __name__ == "__main__":
    main()