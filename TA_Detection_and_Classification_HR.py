import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import joblib
import base64
import os 

# Helper functions
def mean_squared_error(image1, image2):
    return np.mean((image1 - image2) ** 2)

def peak_signal_noise_ratio(image1, image2):
    mse = mean_squared_error(image1, image2)
    return 10 * np.log10(255 ** 2 / mse)

def contrast_enhance(img_clahe, clip_limit):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(15, 15))
    Red = img_clahe[..., 0]
    Green = img_clahe[..., 1]
    Blue = img_clahe[..., 2]
    Green_fix = clahe.apply(Green)
    new_img = np.stack([Red, Green_fix, Blue], axis=2)
    return new_img

def remove_black_background_circular(image):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(center[0], center[1], width - center[0], height - center[1])
    cv2.circle(mask, center, radius, 255, thickness=-1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

def apply_mask(image, mask):
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def process_image(image, clip_limit=1, target_size=(512, 512)):
    original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV BGR format
    original_resize = cv2.resize(original, target_size)
    masked_resize, mask = remove_black_background_circular(original_resize)
    green_channel = original_resize[..., 1]
    clahe_fix = contrast_enhance(original_resize, clip_limit)
    median = cv2.medianBlur(clahe_fix, 3)

    median_masked = apply_mask(median, mask)

    mse_clahe = mean_squared_error(original_resize, clahe_fix)
    psnr_clahe = peak_signal_noise_ratio(original_resize, clahe_fix)

    original_rgb = cv2.cvtColor(original_resize, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    masked_resize_rgb = cv2.cvtColor(masked_resize, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    clahe_fix_rgb = cv2.cvtColor(clahe_fix, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    median_masked_rgb = cv2.cvtColor(median_masked, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display

    return original_rgb, masked_resize_rgb, green_channel, clahe_fix_rgb, median_masked_rgb, mse_clahe, psnr_clahe

def calculate_lbp_histogram(image, radius=3, n_points=24):
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_lbp_features(images, filenames, dataset):
    features_with_filenames = []
    for image, filename in zip(images, filenames):
        lbp_feature = calculate_lbp_histogram(image)
        diagnosis = dataset.loc[dataset['Filename'] == filename, 'Diagnosis'].values[0]
        features_with_filenames.append((filename, lbp_feature, diagnosis))
    return features_with_filenames

# Streamlit app configuration
st.set_page_config(layout="wide")

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path atau URL menuju gambar logo
logo_path1 = 'BME.png'
logo_path2 = 'ITS.png'

# Convert images to base64
logo1_base64 = load_image(logo_path1)
logo2_base64 = load_image(logo_path2)

# Menampilkan logo di kanan atas dan kiri atas
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="data:image/png;base64,{logo2_base64}" width="100">
        </div>
        <div>
            <img src="data:image/png;base64,{logo1_base64}" width="100">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.title('TUGAS AKHIR')
st.title('Deteksi dan Klasifikasi Retinopati Hipertensi dari Citra Fundus Retina menggunakan Support Vector Machine (SVM)')
st.markdown('---')
st.subheader('Nama: Verrent Ulayya Ans')
st.subheader('NRP: 5023201066')
st.subheader('Dosen Pembimbing 1: Dr. Tri Arief Sardjono, S.T., M.T.')
st.subheader('Dosen Pembimbing 2: Nada Fitrieyatul Hikmah, S.T., M.T.')

# Membuat tab di dalam halaman utama
tabs = st.tabs(["Input Data Pasien", "Pemrosesan dan Summary"])

# Fungsi untuk halaman input data pasien
def main():
    with tabs[0]:
        st.header("Input Data Pasien")

        nama_pasien = st.text_input("Nama Pasien")
        id_pasien = st.text_input("ID Pasien")
        umur_pasien = st.number_input("Umur Pasien", min_value=0, max_value=120, step=1)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan", "Lainnya"])

        # Menyimpan data pasien ke session_state
        if st.button("Simpan Data Pasien"):
            st.session_state['data_pasien'] = {
                "Nama Pasien": nama_pasien,
                "ID Pasien": id_pasien,
                "Umur Pasien": umur_pasien,
                "Jenis Kelamin": jenis_kelamin}
            st.success("Data pasien berhasil disimpan!")

def process():
    with tabs[1]:
        st.header("Pemrosesan dan Summary")

        # Upload gambar
        st.subheader('Upload Image')
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        # Upload dataset
        st.subheader('Upload Dataset')
        uploaded_data = st.file_uploader("Upload Dataset", type="xlsx")

        if uploaded_image is not None and uploaded_data is not None:
            image = Image.open(uploaded_image)
            st.session_state['uploaded_image'] = image
            st.image(image, width=150, caption='Uploaded Image', use_column_width=False)

            df = pd.read_excel(uploaded_data)
            st.session_state['uploaded_data'] = df
            st.write('Dataframe dari file yang diunggah:')
            st.write(df)

            # Button to run the processing pipeline
            if st.button("Process"):
                # Process the image
                # Process the image
                # Process the image
                original_rgb, masked_resize_rgb, green_channel, clahe_fix_rgb, median_masked_rgb, mse_clahe, psnr_clahe = process_image(image)
                st.session_state['processed_images'] = {
                    'original': original_rgb,
                    'resize': masked_resize_rgb,
                    'green_channel': green_channel,
                    'clahe_fix': clahe_fix_rgb,
                    'median_masked': median_masked_rgb
                }

                cols = st.columns(5)

                cols[0].image(original_rgb, width=150, caption='Image', use_column_width=False)
                cols[1].image(masked_resize_rgb, width=150, caption='Resize Image', use_column_width=False)
                cols[2].image(green_channel, width=150, caption='Green Channel', use_column_width=False, clamp=True)
                cols[3].image(clahe_fix_rgb, width=150, caption='CLAHE', use_column_width=False)
                cols[4].image(median_masked_rgb, width=150, caption='Median Filter', use_column_width=False)

                # Display evaluation metrics
                #st.write("### Evaluation Metrics")
                #st.write(f"Mean Squared Error (MSE) - CLAHE: {mse_clahe}")
                #st.write(f"Peak Signal-to-Noise Ratio (PSNR) - CLAHE: {psnr_clahe}")

                # LBP Feature Extraction
                lbp_histogram = calculate_lbp_histogram(image)
                st.session_state['lbp_histogram'] = lbp_histogram

                # Display LBP histogram
                st.write("### LBP Histogram")
                plt.figure(figsize=(6, 4))
                plt.bar(range(len(lbp_histogram)), lbp_histogram, width=0.5, color='blue')
                plt.title('LBP Histogram')
                st.session_state['lbp_hist_fig'] = plt

                st.pyplot(plt)

                # Display LBP feature values
                st.write("### LBP Feature Values")

                # Initialize data for the table
                table_data = {
                    "Fitur": [],
                    "Nilai Fitur": [],
                    "Fitur": [],
                    "Nilai Fitur": []
                }

                # Fill data with LBP histogram features
                for i, value in enumerate(lbp_histogram):
                    if i < len(lbp_histogram) // 2:
                        table_data["Fitur"].append(f"Feature {i}")
                        table_data["Nilai Fitur"].append(f"{value:.4f}")
                    else:
                        table_data["Fitur"].append(f"Feature {i}")
                        table_data["Nilai Fitur"].append(f"{value:.4f}")

                # Create a DataFrame from the table data
                table_df = pd.DataFrame(table_data)

                # Display the table
                st.table(table_df)

                # LBP Features with filenames for dataset
                median_images = [median_masked_rgb]  # Menggunakan median_masked_rgb yang sudah dikonversi ke RGB
                filenames = [uploaded_image.name]
                lbp_features_with_filenames = extract_lbp_features(median_images, filenames, df)

                # Inisialisasi list untuk menyimpan fitur LBP dan label diagnosis
                lbp_features = []
                diagnosis_labels = []

                for filename, lbp_feature, diagnosis in lbp_features_with_filenames:
                    lbp_features.append(lbp_feature)
                    diagnosis_labels.append(diagnosis)

                # Preprocess Data for SVM
                x = np.vstack(lbp_features)
                expected_fitur = 26
                if x.shape[1] < expected_fitur:
                    zero_fitur = expected_fitur - x.shape[1]
                    x_zero = np.hstack((x, np.zeros((x.shape[0], zero_fitur))))
                else:
                    x_zero = x

                y = np.array(diagnosis_labels)

                # Convert y to 1D array
                y = y.ravel()

                # Load SVM model from .pkl file
                model_path = 'svm_modelPOLY.pkl'
                svm_model = joblib.load(model_path)

                # Predict using loaded SVM model
                y_pred = svm_model.predict(x_zero)

                # Menampilkan informasi pasien sebelum hasil prediksi
                if 'data_pasien' in st.session_state:
                    data_pasien = st.session_state['data_pasien']
                    st.subheader("Informasi Pasien")
                    st.write(f"**Nama Pasien:** {data_pasien['Nama Pasien']}")
                    st.write(f"**ID Pasien:** {data_pasien['ID Pasien']}")
                    st.write(f"**Umur Pasien:** {data_pasien['Umur Pasien']}")
                    st.write(f"**Jenis Kelamin:** {data_pasien['Jenis Kelamin']}")

                # Print prediction result
                st.write("### Hasil Prediksi:")
                if y_pred[0] == 1:
                    st.write("Diagnosis: Normal")
                else:
                    st.write("Diagnosis: Retinopati Hipertensi")

                st.success("Proses Pre-processing, Ekstraksi Fitur, dan Klasifikasi selesai!")

                # Save prediction result to session_state
                if 'svm_prediction' not in st.session_state:
                    st.session_state['svm_prediction'] = {}

                st.session_state['svm_prediction']['y_pred'] = y_pred[0]

            # Reset button
            if st.button("Reset"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

# Menampilkan halaman input data, proses data, dan summary
main()
process()
