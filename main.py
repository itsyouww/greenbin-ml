import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/model_sampah.h5')
class_names = ['cardboard', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (150, 150)

cap = cv2.VideoCapture(0)
print("🚀 Kamera dijalankan. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal mengambil gambar dari kamera")
        break

    display_frame = frame.copy()

    # Ukuran frame
    h, w, _ = frame.shape
    box_size = 300
    x_center = w // 2
    y_center = h // 2
    x1 = x_center - box_size // 2
    y1 = y_center - box_size // 2
    x2 = x_center + box_size // 2
    y2 = y_center + box_size // 2

    # ROI (Region of Interest)
    roi = frame[y1:y2, x1:x2]
    img = cv2.resize(roi, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    confidence = np.max(prediction)

    # Gambar kotak fokus di tengah
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label hasil prediksi
    label_text = f"{class_label} ({confidence*100:.2f}%)"
    cv2.putText(display_frame, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Klasifikasi Sampah (fokus tengah)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
