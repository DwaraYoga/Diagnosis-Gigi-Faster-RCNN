from flask import Flask
from flask import Flask, render_template, request
import pandas as pd
import cv2
import torch
from inference import run_inference


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# Path ke file model .pth
MODEL_PATH = "models/best_model.pth"


# Default arguments for inference
args = {
    'data': 'config.yaml',
    'weights': MODEL_PATH,  # Replace with your model path
    'threshold': 0.3,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'imgsz': 640,
    'square_img': True,
    'classes': None
}

@app.route('/', methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        img = request.files['file']
        img_path = "static/uploads/" + img.filename
        img.save(img_path)

        # Load deskripsi.xlsx dan buat dictionary
        deskripsi_file = "data/deskripsi.xlsx"
        deskripsi_df = pd.read_excel(deskripsi_file)

        # Asumsikan file Excel memiliki kolom 'class' dan 'description'
        class_to_description = dict(zip(deskripsi_df['Nama Penyakit'], deskripsi_df['Deskripsi']))
        class_to_treatment = dict(zip(deskripsi_df['Nama Penyakit'], deskripsi_df['Perawatan']))

        # Gambar boxes
        output_image, pred_classes = run_inference(img_path, args)
        
        # Simpan hasil
        img_path_pred = "static/predict/" + img.filename
        cv2.imwrite(img_path_pred, output_image)

        pred_classes = list(set(pred_classes))
        pred_descriptions = [class_to_description.get(cls, "Deskripsi tidak ditemukan") for cls in pred_classes]
        pred_treatments = [class_to_treatment.get(cls, "Perawatan tidak ditemukan") for cls in pred_classes]

        # Gabungkan deskripsi dan perawatan menjadi string untuk ditampilkan
        pred_results = {
            cls: {
                "Deskripsi": class_to_description.get(cls, "Deskripsi tidak ditemukan"),
                "Perawatan": class_to_treatment.get(cls, "Perawatan tidak ditemukan")
            }
            for cls in pred_classes
        }

        pred_results_str = '<br>'.join(pred_results)


        return render_template('index.html', 
                            hasil=pred_results,
                            img_pred=img_path_pred,
                            label='')

    return render_template('index.html')

@app.route('/deteksi')
def load_page():
    return render_template('deteksi.html')

@app.route('/test')
def page():
    return render_template('index3.html')

if __name__ == '__main__':
    app.run(debug=True)

