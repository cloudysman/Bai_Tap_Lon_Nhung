import cv2

def capture_images_from_file(image_path, output_prefix, num_images=30):
    original_image = cv2.imread(image_path)
    for i in range(num_images):
        img_name = f"{output_prefix}_{i}.jpg"
        cv2.imwrite(img_name, original_image)
        print(f"{img_name} saved!")

# Đường dẫn tới ảnh gốc của Obama
image_path = 'Obama_1.jpg'
capture_images_from_file(image_path, 'obama')
