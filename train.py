import cv2
import os
import dlib
import numpy as np

# Chụp lại ảnh từ một ảnh gốc và lưu vào thư mục "training"
def capture_images_from_file(image_path, output_folder, num_images=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Unable to read image from {image_path}")
        return
    
    for i in range(num_images):
        img_name = os.path.join(output_folder, f"obama_{i}.jpg")
        cv2.imwrite(img_name, original_image)
        print(f"{img_name} saved!")

# Đường dẫn tới ảnh gốc của Obama
image_path = 'C:/Users/trong/Documents/BTL_Nhung/Obama_1.jpg'
output_folder = 'training'
capture_images_from_file(image_path, output_folder)

# Đường dẫn tới các tệp đã giải nén
shape_predictor_path = r"C:/Users/trong/Documents/BTL_Nhung/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = r"C:/Users/trong/Documents/BTL_Nhung/dlib_face_recognition_resnet_model_v1.dat"

# Trích xuất các đặc trưng của khuôn mặt từ các ảnh trong thư mục "training"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

def extract_face_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_descriptors = []

    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        face_descriptors.append(np.array(face_descriptor))
    
    return face_descriptors

features = []
for i in range(30):
    img_path = os.path.join(output_folder, f"obama_{i}.jpg")
    features.extend(extract_face_features(img_path))

features = np.array(features)
np.save("obama_features.npy", features)
