import cv2
import os
import dlib
import numpy as np

# Đường dẫn tới các tệp đã giải nén
shape_predictor_path = r"C:/Users/trong/Documents/BTL_Nhung/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = r"C:/Users/trong/Documents/BTL_Nhung/dlib_face_recognition_resnet_model_v1.dat"

# Định nghĩa các mô hình Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Hàm trích xuất đặc trưng khuôn mặt
def extract_face_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_descriptors = []
    shapes = []

    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        face_descriptors.append(np.array(face_descriptor))
        shapes.append((face.left(), face.top(), face.right(), face.bottom()))
    
    return face_descriptors, shapes

# Hàm so sánh khuôn mặt
def compare_faces(face_descriptor, known_face_descriptors, threshold=0.6):
    for known_face in known_face_descriptors:
        dist = np.linalg.norm(face_descriptor - known_face)
        if dist < threshold:
            return True
    return False

# Hàm nhận diện khuôn mặt từ ảnh mới và vẽ bounding box
def recognize_and_draw_faces(image_path, known_face_descriptors):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return None

    face_descriptors, shapes = extract_face_features(image)
    for i, face_descriptor in enumerate(face_descriptors):
        if compare_faces(face_descriptor, known_face_descriptors):
            (left, top, right, bottom) = shapes[i]
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, "Obama", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Tải các đặc trưng khuôn mặt đã lưu
known_face_descriptors = np.load("obama_features.npy")

# Đường dẫn tới ảnh mới của Obama
new_image_path = 'C:/Users/trong/Documents/BTL_Nhung/Obama_new.jpg'

# Nhận diện và vẽ bounding box
result_image = recognize_and_draw_faces(new_image_path, known_face_descriptors)

if result_image is not None:
    # Lưu kết quả ra file
    result_path = "recognized_obama.jpg"
    cv2.imwrite(result_path, result_image)
    print(f"Recognition result saved to {result_path}")
else:
    print("Recognition failed.")
