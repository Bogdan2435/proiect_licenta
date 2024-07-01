import cv2
import os
import csv
import math

def calculate_angle(x, y):
    center_x = 320  
    center_y = 480  
    dx = x - center_x
    dy = center_y - y  
    return math.degrees(math.atan2(dy, dx))

def annotate_images(image_folder, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image Name', 'X', 'Y', 'Angle', 'Complementary Angle'])

        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        
        for image_file in image_files:

            if image_file.endswith('.png'):
                image_path = os.path.join(image_folder, image_file)
                image = cv2.imread(image_path)
                image_copy = image.copy()

                cv2.imshow('Image', image_copy)

                def click_event(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
                        cv2.imshow('Image', image_copy)

                        angle = calculate_angle(x, y)
                        complementary_angle = 180 - angle

                        csvwriter.writerow([image_file, x, y, angle, complementary_angle])

                cv2.setMouseCallback('Image', click_event)

                cv2.waitKey(0)

                cv2.destroyAllWindows()

if __name__ == "__main__":
    image_folder = 'poze_pista_prezentare/select_poze'
    csv_filename = 'poze_pista_prezentare/csv/adnotari_noi.csv'
    annotate_images(image_folder, csv_filename)
