import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import cv2
import os
import torch.nn as nn
import torchvision.models as models
import time
import requests
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import threading
import queue

live_stream_url = "http://172.20.10.2:5000"

default_speed = 0.40
default_turn_gain = 0.30
default_offset = 0.02 

crosswalk_speed = 0.3
crosswalk_turn_gain = 0.15

class MobileNetSteering(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetSteering, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.mobilenet(x)
        return x

# Functia de preprocesare a cadrelor pentru modelul de predictie a unghiului de virare, convertind imaginea in formatul YUV si redimensionand-o la dimensiunea de 200x66
def img_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.resize(image, (200, 66))
    return image

# Clasa pentru citirea frameurilor dintr-un stream video live (stackoverflow)
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Citesc frameurile imediat ce sunt disponibile, pastrand doar cel mai recent frame
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() 
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self): # returnarea frameului curent
        return self.q.get()
    
def predict_steering_angle(frame, model, transform):

    # cv2.imwrite("poze_mama_24_iunie/" + str("{:.5f}".format(time.time())) + ".png", frame)
    image = img_preprocess(frame)
    image = transform(image)

    image = image.unsqueeze(0)
    image = image.to(device)        

    with torch.no_grad():
        time1 = time.time()
        prediction = model(image)
        time2 = time.time()
        # Optional pot printa timpul de inferenta
        # print(f'Inference Time: {(time2 - time1) * 1000:.2f} ms')
    predicted_steering_angle = float(prediction.float()[0])

    # pot salva frame ul cu predictia pentru virare
    # cv2.putText(frame, f'Steering Angle: {predicted_steering_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imwrite("poze_mama_24_iunie/predicted/" + str("{:.5f}".format(time.time())) + ".png", frame)
   
    return predicted_steering_angle

# Functie pentru controlul robotului, in care se calculeaza puterea motorului stang si drept in functie de unghiul de virare si viteza + trimiterea comenzii catre robot
def control_robot(steering_angle, speed = 0.4, turn_gain = 0.2, offset = 0.02):
    center = (90 - steering_angle) / 90
    left = float((speed - turn_gain * center - offset)) 
    right = float((speed + turn_gain * center) + offset)

    # Optional pot printa puterea calculata pentru motoare
    # print(f'Steering Angle: {steering_angle}, Left: {left}, Right: {right}')

    url = live_stream_url + '/move_robot'  
    payload = {'left': left, 'right': right}
    response = requests.post(url, json=payload)

    # Optional pot printa raspunsul primit de la robot daca comanda a fost trimisa cu succes
    # if response.status_code == 200:
    #     print('Command sent successfully')
    # else:
    #     print('Failed to send command')

def capture_frame_and_control_robot(model, transform, yolo):
    cap = VideoCapture(live_stream_url + '/video_feed') # Initializez metoda de citire a frameurilor din stream cu url-ul de unde se poate accesa stream-ul video

    # Pentru detectia semnelor de circulatie, initializez variabilele de stare ca fiind False (nu sunt detectate)
    state_stop = False
    state_left = False
    state_right = False
    state_crosswalk = False
    state_left = False
    state_right = False

    # Variabila pentru offsetul default dintre puterea motorului stang si cel drept
    offset_value = default_offset

    while True:
        start_time = time.time()

        frame = cap.read()

        # daca am detectat semnul de stanga sau dreapta, setez offsetul dintre motoare corespunzator ca robotul sa vireze
        if state_left or state_right: 
            if state_left:
                offset = 0.13
                if time.time() - timp_start_left > 0.5:
                    offset = offset_value
            if state_right:
                offset = -0.07
                if time.time() - timp_start_right > 0.3:
                    offset = offset_value
        else: # daca semnele de directie nu sunt detectate, offsetul revine la cel default
            offset = offset_value

        if state_left and time.time() - timp_start_left >= 1:
            state_left = False
        if state_right and time.time() - timp_start_right >= 1:
            state_right = False
        if state_stop and time.time() - state_stop_time >= 1:
            state_stop = False
            state_stop_time = 0

        result = yolo.predict(source=frame, device='mps')[0]
            
        
        # Daca detectez indicatorul de trecere de pietoni, reduc viteza si setez turn_gain-ul corespunzator vitezei reduse
        if state_crosswalk:
            speed = crosswalk_speed
            turn_gain = crosswalk_turn_gain
            if time.time() - timp_start_cw > 1: # daca a trecut o secunda de cand a fost detectata trecerea
                state_crosswalk = False
        else: # daca nu detectez trecerea de pietoni, robotul merge cu viteza normala si turn_gain-ul default
            speed = default_speed
            turn_gain = default_turn_gain
        
        steering_angle = predict_steering_angle(frame, model, transform) # Predictia unghiului de virare 

        control_robot(steering_angle=steering_angle, speed=speed, turn_gain=turn_gain, offset = offset) 

        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        print(f'Processing Time: {processing_time_ms:.2f} ms')

        if state_stop == False and state_left == False and state_right == False:

            boxes = []
            annotator = Annotator(frame)
            for label in result.boxes.data:        

                # Optional pot salva frameurile cu semnele de circulatie detectate pentru a verifica daca detectiile sunt corecte
                # annotator.box_label(
                #     label[0:4],
                #     f"{result.names[label[-1].item()]} {label[-2]}",
                # color_list[round(label[-1].item())])
                # cv2.imwrite("poze_adn/" + str("{:.5f}".format(time.time())) + ".png", annotator.im)

                box = [float(lbl) for lbl in label[0:4]]
                label_text = result.names[label[-1].item()]
                confidence = float(label[-2])
                
                # Verific pentru fiecare caz 
                if label_text == 'stop' and confidence >= 0.4 and state_stop == False:
                    # Daca detectez semnul de stop si confidenta este mai mare de 0.4 si aria semnului este mai mare de 2000 pixeli atunci atunci opresc robotul pentru 3 secunde 
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    if width * height >= 2000:
                        # print("===============================")
                        # print("Stop detectat")
                        
                        url = live_stream_url + '/move_robot' 
                        payload = {'left': 0, 'right': 0}
                        response = requests.post(url, json=payload)
                        # if response.status_code == 200:
                        #     print('Robot oprit la stop')
                        # else:
                        #     print('Failed to send command')
                        time.sleep(3)
                        state_stop = True
                        state_stop_time = time.time()

                if label_text == 'red' and confidence >= 0.4: 
                    # Daca detectez semnul de oprire la semafor si confidenta este mai mare de 0.4 si aria semnului este mai mare de 500 pixeli atunci opresc robotul
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    if width * height >= 500:
                        print("===============================")
                        print("Rosu detectat")
                        state_Red = True
                        
                        url = live_stream_url + '/move_robot'
                        payload = {'left': 0, 'right': 0}
                        response = requests.post(url, json=payload)
                        if response.status_code == 200:
                            print('Robot oprit la rosu')
                            time.sleep(0.5)
                        else:
                            print('Failed to send command')

                if label_text == 'green':
                    state_Red = False
                    print("Verde detectat")

                if label_text == 'crosswalk' and confidence >= 0.4 and state_crosswalk == False:
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    if width * height >= 100:
                        state_crosswalk = True
                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++TRECERE DE PIETONI DETECTATA")
                        timp_start_cw = time.time()

                if label_text == 'left' and confidence >= 0.6 and state_left == False:
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    if width * height >= 700:
                        state_left = True
                        print("Stanga Detectat")
                        timp_start_left = time.time()

                if label_text == 'right' and confidence >= 0.6 and state_right == False:
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    if width * height >= 700:
                        state_right = True
                        print("Dreapta Detectat")
                        timp_start_right = time.time()

                print("Detected " + str(label_text) + " at " + str(box) + " with confidence: " + str(confidence))

color_list = [(0,0,255), (255,0,0), (0,255,0),  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

if __name__ == '__main__':
    model = MobileNetSteering()
    device = torch.device("cpu")
    model_yolo = YOLO('Modele/yolov8n_srg_cross_left_right_8_iunie.pt')

    checkpoint = torch.load('Modele/mobilenet_steering_8iunie.pth', map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),           
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    try:
        capture_frame_and_control_robot(model=model, transform=transform, yolo=model_yolo)
    except KeyboardInterrupt:
        # Cand opresc script-ul, opresc si robotul
        url = live_stream_url + '/move_robot' 
        payload = {'left': 0, 'right': 0}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print('Command sent successfully')
        else:
            print('Failed to send command')

