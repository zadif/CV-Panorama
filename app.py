import gradio as gr
import cv2
import numpy as np
import spaces

from orbSift import extractFeatures, matchFeatures, computeHomography, warpAndStitch

@spaces.GPU
def process2Images(img1, img2, method):
    if img1 is None or img2 is None:
        raise gr.Error("Please upload both Image 1 and Image 2.")
    
    try:
        bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        kp1, des1, kp2, des2 = extractFeatures(bgr1, bgr2, method=method)

        goodMatches = matchFeatures(des1, des2, method=method)

        H, mask = computeHomography(kp1, kp2, goodMatches)

        panorama = warpAndStitch(bgr1, bgr2, H)

        return cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

    except Exception as e:
        raise gr.Error(f"Custom Stitching Error: {str(e)}")


@spaces.GPU
def processManyImages(fileList):
    if not fileList or len(fileList) < 2:
        raise gr.Error("Please upload at least 2 images.")
    
    try:
        images = []
        for file in fileList:
            img = cv2.imread(file.name)
            if img is not None:
                images.append(img)

        if len(images) < 2:
            raise gr.Error("Failed to load valid images.")

        stitcher = cv2.Stitcher_create()
        status, panorama = stitcher.stitch(images)

        if status != cv2.Stitcher_OK:
            raise gr.Error(f"OpenCV Stitcher failed with status code: {status}")

        return cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

    except Exception as e:
        raise gr.Error(f"Multi-Image Stitching Error: {str(e)}")



with gr.Blocks(title="Computer Vision Panorama App") as demo:
    gr.Markdown("# Computer Vision Panorama Stitcher")
    gr.Markdown("Select a mode below to stitch your photos.")

    with gr.Tabs():
        # =========================================================
        # BOX 2: MERGE MANY IMAGES (OpenCV Production Engine)
        # =========================================================
        with gr.TabItem("Merge Many Images"):
            gr.Markdown("### Open CV Stitcher")
            
            # Upload Multiple Files
            input_files = gr.File(file_count="multiple", file_types=["image"], label="Upload 2 or More Photos")
            
            btn_many = gr.Button("Stitch Multiple Images", variant="primary")
            output_many = gr.Image(label="Multi-Image Panorama Result")
            
            # Button Click Event
            btn_many.click(
                fn=processManyImages,
                inputs=input_files,
                outputs=output_many
            )
        # =========================================================
        # BOX 1: MERGE 2 IMAGES (Custom SIFT / ORB)
        # =========================================================
        with gr.TabItem("Merge 2 Images (SIFT / ORB)"):
            gr.Markdown("### Feature Detection + RANSAC + Homography Pipeline")
            
            # Select Algorithm
            method_radio = gr.Radio(["SIFT", "ORB"], value="SIFT", label="Select Feature Algorithm")
            
            # Upload 2 Images
            with gr.Row():
                input_img1 = gr.Image(label="Image 1 (Left)")
                input_img2 = gr.Image(label="Image 2 (Right)")
            
            btn_2img = gr.Button("Stitch 2 Images", variant="primary")
            output_2img = gr.Image(label="Panorama Result")
            
            # Button Click Event
            btn_2img.click(
                fn=process2Images,
                inputs=[input_img1, input_img2, method_radio],
                outputs=output_2img
            )

        

# Launch the app
if __name__ == "__main__":
    demo.launch()