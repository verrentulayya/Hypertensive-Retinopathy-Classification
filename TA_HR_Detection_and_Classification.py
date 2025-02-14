import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import joblib
import os

# Helper functions
def mean_squared_error(image1, image2):
    return np.mean((np.array(image1) - np.array(image2)) ** 2)

def peak_signal_noise_ratio(image1, image2):
    mse = mean_squared_error(image1, image2)
    return 10 * np.log10(255 ** 2 / mse)

def contrast_enhance(img_clahe, clip_limit):
    img = ImageEnhance.Contrast(img_clahe)
    img = img.enhance(clip_limit)
    return img

def remove_black_background_circular(image):
    np_image = np.array(image)
    height, width = np_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(center[0], center[1], width - center[0], height - center[1])
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask[dist_from_center <= radius] = 255
    masked_image = np_image * (mask[:, :, np.newaxis] // 255)
    masked_image_pil = Image.fromarray(masked_image)
    return masked_image_pil, mask

def apply_mask(image, mask):
    np_image = np.array(image)
    masked_image = np_image * (mask[:, :, np.newaxis] // 255)
    masked_image_pil = Image.fromarray(masked_image)
    return masked_image_pil

def process_image(image, clip_limit=1, target_size=(512, 512)):
    original = ImageOps.exif_transpose(image)
    original_resize = original.resize(target_size)
    masked_resize, mask = remove_black_background_circular(original_resize)
    green_channel = np.array(original_resize)[:, :, 1]
    clahe_fix = contrast_enhance(original_resize, clip_limit)
    median = clahe_fix.filter(ImageFilter.MedianFilter(size=3))

    median_masked = apply_mask(median, mask)

    mse_clahe = mean_squared_error(original_resize, clahe_fix)
    psnr_clahe = peak_signal_noise_ratio(original_resize, clahe_fix)

    return original_resize, masked_resize, green_channel, clahe_fix, median_masked, mse_clahe, psnr_clahe

def calculate_lbp_histogram(image, radius=3, n_points=24):
    image_gray = ImageOps.grayscale(image)
    np_image_gray = np.array(image_gray)
    lbp = local_binary_pattern(np_image_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_lbp_features(images):
    features = []
    for image in images:
        lbp_feature = calculate_lbp_histogram(image)
        features.append(lbp_feature)
    return features

# Streamlit app configuration
st.set_page_config(layout="wide")
st.title('Deteksi dan Klasifikasi Retinopati Hipertensi dari Citra Fundus Retina menggunakan Support Vector Machine (SVM)')
st.markdown('---')
st.subheader('Nama: Verrent Ulayya Ans')
st.subheader('NRP: 5023201066')

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

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.session_state['uploaded_image'] = image
            st.image(image, width=150, caption='Uploaded Image', use_column_width=False)

            # Button to run the processing pipeline
            if st.button("Process"):
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
                }

                # Fill data with LBP histogram features
                for i, value in enumerate(lbp_histogram):
                    table_data["Fitur"].append(f"Feature {i}")
                    table_data["Nilai Fitur"].append(f"{value:.4f}")

                # Create a DataFrame from the table data
                table_df = pd.DataFrame(table_data)

                # Display the table
                st.table(table_df)

                # LBP Features for the uploaded image
                median_images = [median_masked_rgb]  # Using median_masked_rgb that has been converted to RGB
                lbp_features = extract_lbp_features(median_images)

                # Preprocess Data for SVM
                x = np.vstack(lbp_features)
                expected_fitur = 26
                if x.shape[1] < expected_fitur:
                    zero_fitur = expected_fitur - x.shape[1]
                    x_zero = np.hstack((x, np.zeros((x.shape[0], zero_fitur))))
                else:
                    x_zero = x

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
                if y_pred[0] == 0:
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
