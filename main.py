from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
import cv2 as cv
import numpy as np
from orbSift import decodeImage, extractFeatures, matchFeatures, computeHomography, warpAndStitch

app=FastAPI()


@app.get("/")
def health():
    return{"message":"Fast API ki khidmt sy salam arz krty hain"}


@app.post("/stitch")
async def stitch_images(
    files: List[UploadFile] = File(...),
    method: str = Form("SIFT") 
):
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Must upload exactly 2 images")

    try:

        img1 = decodeImage(await files[0].read())
        img2 = decodeImage(await files[1].read())

        kp1, des1, kp2, des2 = extractFeatures(img1, img2, method=method)

        goodMatches = matchFeatures(des1, des2, method=method)

        H, mask = computeHomography(kp1, kp2, goodMatches)

        panorama = warpAndStitch(img1, img2, H)

        _, encodedImg = cv.imencode(".jpg", panorama)
        return Response(content=encodedImg.tobytes(), media_type="image/jpeg")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}") 
    




@app.post("/cv-stitch")
async def cvStitcher(files: List[UploadFile] = File(...)):

    if len(files)<2:
        raise HTTPException(status_code=400, detail="Must upload at least 2 images")

    try:
        images=[]
        for file in files:
            images.append(decodeImage(await file.read())) 

        stitcher = cv.Stitcher_create()
        status, panorama = stitcher.stitch(images)

        if status != cv.Stitcher_OK:
            raise ValueError(f"OpenCV Stitcher failed with status code: {status}")

        _, encodedImg = cv.imencode(".jpg", panorama)

        return Response(content=encodedImg.tobytes(), media_type="image/jpeg")



    except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}") 


