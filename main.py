from fastapi import FastAPI, Response
from ImageProcessor.ImageProcessor import ImageProcessor
from fastapi.responses import StreamingResponse
import uvicorn

image_processor = ImageProcessor()

# API 
app = FastAPI()

def generate_frames(view):
    while True:
        if (view == 1):
            frame_bytes = image_processor.get_frame()
        else:
            frame_bytes = image_processor.get_aile_view()
        if frame_bytes is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def get_heatmap():
    frame_bytes = image_processor.get_heatmap()
    if frame_bytes is None:
        return None
    yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    # return image_processor.get_heatmap()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(1), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/aisle_view")
async def aisle_view():
    return StreamingResponse(generate_frames(2), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/get_heatmap")
async def get_heatmap():
    # return image_processor.get_heatmap as encoded image

    result = image_processor.get_heatmap()
    return Response(content=result, media_type="image/jpeg")

@app.get("/get_trajectories")
async def get_trajectories():
    result = image_processor.get_common_paths()
    return Response(content=result, media_type="image/jpeg")

@app.get("/")
def read_root():
    return {"Message": "TCC-API"}

# Entry point to run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
