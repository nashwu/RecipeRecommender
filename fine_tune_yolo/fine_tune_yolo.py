from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("runs/train7/weights/last.pt")

    model.train(data = 'dataset/data.yaml', imgsz = 640, epochs = 100, batch = 16, 
                device = 'cuda', project = 'runs', resume=True)