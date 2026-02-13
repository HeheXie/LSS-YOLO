import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml')
    model.train(data='C:/Users/XieHe/Desktop/ultralytics-main/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=32,
                close_mosaic=1,
                workers=8, 
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, 
                # resume=True,
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )