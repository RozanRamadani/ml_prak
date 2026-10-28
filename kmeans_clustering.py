import os
import sys
import tempfile
import warnings

# Configure environment variables to avoid matplotlib/loky issues on Windows
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 1. IMPORT LIBRARY (Sesuai Modul 13 Bagian 1)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_kmeans():
    print("=" * 70)
    print("PRAKTIKUM MACHINE LEARNING - CLUSTERING DENGAN K-MEANS")
    print("Dataset: Student Performance Factors")
    print("=" * 70)

    # Set output directory
    output_dir = os.path.join("outputs", "kmeans")
    os.makedirs(output_dir, exist_ok=True)

    # Set plot style (Modul 13 style)
    try:
        plt.style.use('fivethirtyeight')
    except Exception:
        plt.style.use('default')

    # ==========================================
    # 2. IMPORT DATA (Sesuai Modul 13 Bagian 2)
    # ==========================================
    dataset_path = os.path.join("data", "StudentPerformanceFactors.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {dataset_path}")

    print(f"\n[LANGKAH 2] Membaca dataset dari: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print("5 Data Teratas (df.head()):")
    print(df.head())

    # ==========================================
    # 3. AMATI BENTUK DATA (Sesuai Modul 13 Bagian 3)
    # ==========================================
    print(f"\n[LANGKAH 3] Bentuk Data (df.shape): {df.shape}")
    print(f"Jumlah baris (observasi): {df.shape[0]}")
    print(f"Jumlah kolom (fitur): {df.shape[1]}")

    # ==========================================
    # 4. STATISTIK DESKRIPTIF (Sesuai Modul 13 Bagian 4)
    # ==========================================
    print("\n[LANGKAH 4] Ringkasan Statistik Deskriptif (df.describe()):")
    num_summary = df.describe().T
    print(num_summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

    # ==========================================
    # 5. CEK NULL DATA (Sesuai Modul 13 Bagian 5)
    # ==========================================
    print("\n[LANGKAH 5] Pemeriksaan Missing Values (df.isnull().sum()):")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print("Kolom dengan nilai kosong:")
        print(missing_cols)
    else:
        print("Tidak ditemukan missing values pada seluruh kolom.")

    # Cek duplikasi
    duplicates = df.duplicated().sum()
    print(f"Jumlah baris duplikat: {duplicates}")

    # ==========================================
    # 6. CEK OUTLIER (Sesuai Modul 13 Bagian 6)
    # ==========================================
    print("\n[LANGKAH 6] Pemeriksaan Outlier Menggunakan Metode IQR:")
    num_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
    outlier_summary = {}
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_summary[col] = {
            'Q1': q1, 'Q3': q3, 'IQR': iqr, 
            'Lower Bound': lower_bound, 'Upper Bound': upper_bound,
            'Outliers': outliers_count, 'Outlier %': (outliers_count / len(df)) * 100
        }
    outlier_df = pd.DataFrame(outlier_summary).T
    print(outlier_df[['Lower Bound', 'Upper Bound', 'Outliers', 'Outlier %']])

    # ==========================================
    # 7. AMATI BENTUK VISUAL MASING-MASING FITUR (Sesuai Modul 13 Bagian 7)
    # ==========================================
    print("\n[LANGKAH 7] Membuat Visualisasi Distribusi Fitur...")
    plt.figure(1, figsize=(15, 6))
    features_to_plot = ['Hours_Studied', 'Attendance', 'Previous_Scores']
    n = 0
    for x in features_to_plot:
        n += 1
        plt.subplot(1, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        sns.histplot(df[x], kde=True, stat="density", kde_kws=dict(cut=3), bins=20, color='royalblue')
        plt.title(f'Distplot of {x}', fontsize=12)
        plt.xlabel(x)
        plt.ylabel('Density')
    
    dist_plot_path = os.path.join(output_dir, "01_feature_distributions.png")
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar distribusi tersimpan di: {dist_plot_path}")

    # ==========================================
    # 8. PLOTTING RELASI ANTAR FITUR (Sesuai Modul 13 Bagian 8)
    # ==========================================
    print("\n[LANGKAH 8] Membuat Visualisasi Relasi Antar Fitur (sns.regplot)...")
    plt.figure(2, figsize=(15, 15))
    n = 0
    for x in features_to_plot:
        for y in features_to_plot:
            n += 1
            plt.subplot(3, 3, n)
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
            sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha': 0.2, 's': 15}, line_kws={'color': 'crimson'})
            plt.title(f'{y} vs {x}', fontsize=10)
    
    rel_plot_path = os.path.join(output_dir, "02_feature_relationships.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar relasi antar-fitur tersimpan di: {rel_plot_path}")

    # ==========================================
    # 9. SEBARAN BERDASARKAN KATEGORI (Sesuai Modul 13 Bagian 9)
    # ==========================================
    print("\n[LANGKAH 9] Visualisasi Sebaran Berdasarkan Kategori Gender...")
    plt.figure(3, figsize=(12, 6))
    for gender in ['Male', 'Female']:
        subset = df[df['Gender'] == gender]
        plt.scatter(
            x=subset['Hours_Studied'], 
            y=subset['Attendance'], 
            s=40, 
            alpha=0.4, 
            label=gender
        )
    plt.xlabel('Hours_Studied', fontsize=12)
    plt.ylabel('Attendance (%)', fontsize=12)
    plt.title('Hours Studied vs Attendance by Gender', fontsize=14)
    plt.legend()
    gender_plot_path = os.path.join(output_dir, "03_distribution_by_gender.png")
    plt.savefig(gender_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar sebaran gender tersimpan di: {gender_plot_path}")

    # ==========================================
    # 10. ELBOW METHOD UNTUK MENENTUKAN K (Sesuai Modul 13 Bagian 10)
    # ==========================================
    # Pemilihan fitur:
    # Menggunakan 'Hours_Studied' dan 'Attendance' yang merupakan dua faktor komitmen belajar
    # primer siswa. 'Exam_Score' sengaja TIDAK digunakan sebagai fitur clustering agar tidak terjadi
    # target leakage dan memungkinkan pengujian performa ujian pasca-clustering secara objektif.
    print("\n[LANGKAH 10] Menghitung Elbow Method (k=1..10)...")
    cluster_features = ['Hours_Studied', 'Attendance']
    X_raw = df[cluster_features].values

    # Standardisasi fitur (karena rentang Hours_Studied [1-44] dan Attendance [60-100] berbeda jauh)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    inertia = []
    k_range = range(1, 11)
    for n_clusters in k_range:
        algorithm = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=111)
        algorithm.fit(X_scaled)
        inertia.append(algorithm.inertia_)
        print(f"k = {n_clusters:2d} -> Inertia = {algorithm.inertia_:.2f}")

    # Visualisasi kurva Elbow
    plt.figure(4, figsize=(12, 6))
    plt.plot(k_range, inertia, 'o', color='darkblue', markersize=8)
    plt.plot(k_range, inertia, '-', color='royalblue', alpha=0.7, linewidth=2)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method For Optimal k (Student Performance Factors)', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.5)

    elbow_plot_path = os.path.join(output_dir, "04_elbow_curve.png")
    plt.savefig(elbow_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Kurva Elbow tersimpan di: {elbow_plot_path}")

    # ==========================================
    # 11. MEMBANGUN MODEL K-MEANS (Sesuai Modul 13 Bagian 11)
    # ==========================================
    # Berdasarkan kurva elbow dan penurunan inertia yang mulai melandai secara konsisten,
    # serta kemudahan interpretasi profil siswa, k = 3 dipilih sebagai jumlah cluster optimal
    # (Cluster Rendah, Sedang, Tinggi). Evaluasi k=4 juga didokumentasikan untuk perbandingan.
    optimal_k = 3
    print(f"\n[LANGKAH 11] Membangun model KMeans dengan optimal k = {optimal_k}...")
    algorithm = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111)
    algorithm.fit(X_scaled)
    labels = algorithm.labels_
    centroids_scaled = algorithm.cluster_centers_
    # Kembalikan koordinat centroid ke skala asli untuk kemudahan interpretasi
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Simpan label ke salinan DataFrame (tidak mengubah data asli)
    df_clustered = df.copy()
    df_clustered['KMeans_Cluster'] = labels

    print("Centroid dalam skala asli [Hours_Studied, Attendance]:")
    for i, c in enumerate(centroids_original):
        print(f"Cluster {i}: Hours_Studied = {c[0]:.2f} jam, Attendance = {c[1]:.2f}%")

    # ==========================================
    # 12. MENYIAPKAN DATA VISUALISASI BOUNDARY (Sesuai Modul 13 Bagian 12)
    # ==========================================
    print("\n[LANGKAH 12] Menyiapkan meshgrid decision boundary (sesuai Modul 13)...")
    step = 0.05
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # ==========================================
    # 13. VISUALISASI HASIL CLUSTER (Sesuai Modul 13 Bagian 13)
    # ==========================================
    print("\n[LANGKAH 13] Membuat Visualisasi Hasil Clustering...")
    # Plot 1: Sesuai struktur persis Modul 13 (Decision Boundary dengan imshow)
    plt.figure(5, figsize=(14, 7))
    plt.clf()
    # Mengonversi batas extent kembali ke koordinat asli untuk plotting yang informatif
    x_bounds_orig = scaler.inverse_transform([[x_min, 0], [x_max, 0]])[:, 0]
    y_bounds_orig = scaler.inverse_transform([[0, y_min], [0, y_max]])[:, 1]
    
    plt.imshow(
        Z, 
        interpolation='nearest',
        extent=(x_bounds_orig[0], x_bounds_orig[1], y_bounds_orig[0], y_bounds_orig[1]),
        cmap=plt.cm.Pastel2, 
        aspect='auto', 
        origin='lower'
    )
    plt.scatter(
        df['Hours_Studied'], 
        df['Attendance'], 
        c=labels, 
        cmap='viridis', 
        s=30, 
        alpha=0.6,
        edgecolor='k',
        linewidth=0.3
    )
    plt.scatter(
        centroids_original[:, 0], 
        centroids_original[:, 1], 
        s=250, 
        c='red', 
        marker='X',
        label='Centroids'
    )
    plt.title(f'K-Means Clustering Decision Boundary (k={optimal_k})', fontsize=14)
    plt.xlabel('Hours_Studied (Jam Belajar)', fontsize=12)
    plt.ylabel('Attendance (Kehadiran %)', fontsize=12)
    plt.legend(loc='upper left')

    boundary_plot_path = os.path.join(output_dir, "05_kmeans_clusters_decision_boundary.png")
    plt.savefig(boundary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar decision boundary tersimpan di: {boundary_plot_path}")

    # Plot 2: Scatter plot interpretatif dengan label profil
    plt.figure(6, figsize=(12, 7))
    cluster_names = {
        0: 'Cluster 0: Moderat (Average Engagement)',
        1: 'Cluster 1: Rendah (Low Engagement/At-Risk)',
        2: 'Cluster 2: Tinggi (High Engagement/Diligent)'
    }
    # Urutkan penamaan berdasarkan rata-rata Hours_Studied
    cluster_means_hs = df_clustered.groupby('KMeans_Cluster')['Hours_Studied'].mean().to_dict()
    sorted_clusters = sorted(cluster_means_hs.keys(), key=lambda c: cluster_means_hs[c])
    label_map = {
        sorted_clusters[0]: 'Rendah (Low Study & Low Attendance)',
        sorted_clusters[1]: 'Menengah (Moderate Study & Attendance)',
        sorted_clusters[2]: 'Tinggi (High Study & High Attendance)'
    }
    df_clustered['Cluster_Label'] = df_clustered['KMeans_Cluster'].map(label_map)

    palette = {'Rendah (Low Study & Low Attendance)': '#e74c3c',
               'Menengah (Moderate Study & Attendance)': '#3498db',
               'Tinggi (High Study & High Attendance)': '#2ecc71'}

    sns.scatterplot(
        data=df_clustered,
        x='Hours_Studied',
        y='Attendance',
        hue='Cluster_Label',
        palette=palette,
        alpha=0.6,
        s=40
    )
    plt.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        s=200,
        c='black',
        marker='P',
        label='Centroids'
    )
    plt.title(f'Hasil Clustering Siswa Berdasarkan Komitmen Belajar (K-Means k={optimal_k})', fontsize=14)
    plt.xlabel('Hours Studied (Jam Belajar per Minggu)', fontsize=12)
    plt.ylabel('Attendance (Tingkat Kehadiran %)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    scatter_plot_path = os.path.join(output_dir, "06_kmeans_clusters_scatter.png")
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar scatter plot tersimpan di: {scatter_plot_path}")

    # ==========================================
    # 14. SILHOUETTE SCORE (Sesuai Modul 13 Bagian 14)
    # ==========================================
    print("\n[LANGKAH 14] Menghitung Nilai Silhouette Score...")
    score_k3 = silhouette_score(X_scaled, labels)
    print(f">> Silhouette Score untuk k={optimal_k}: {score_k3:.4f}")

    # Bandingkan juga dengan k=2, 4, 5 untuk konteks laporan
    sil_scores = {}
    for test_k in [2, 3, 4, 5, 6]:
        test_algo = KMeans(n_clusters=test_k, init='k-means++', n_init=10, random_state=111)
        test_labels = test_algo.fit_predict(X_scaled)
        s_val = silhouette_score(X_scaled, test_labels)
        sil_scores[test_k] = s_val
        print(f"Silhouette Score (k={test_k}): {s_val:.4f}")

    # ==========================================
    # 15. PROFILING & RINGKASAN CLUSTER
    # ==========================================
    print("\n[LANGKAH 15] Karakteristik & Profil Statistik per Cluster:")
    profile_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Exam_Score']
    cluster_profile = df_clustered.groupby('KMeans_Cluster')[profile_features].agg(['count', 'mean', 'std', 'min', 'max'])
    print(cluster_profile)

    # Simpan ringkasan profil ke CSV
    summary_path = os.path.join(output_dir, "kmeans_cluster_summary.csv")
    cluster_profile.to_csv(summary_path)
    print(f"Ringkasan statistik cluster tersimpan di: {summary_path}")

    # Simpan hasil perbandingan evaluasi ke CSV
    eval_df = pd.DataFrame({
        'k': list(k_range),
        'Inertia': inertia,
        'Silhouette_Score': [None] + [sil_scores.get(k, np.nan) for k in range(2, 11)]
    })
    eval_csv_path = os.path.join(output_dir, "kmeans_evaluation_metrics.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"Metrik evaluasi tersimpan di: {eval_csv_path}")

    # Distribusi jumlah anggota per cluster
    print("\nDistribusi Ukuran Cluster:")
    size_series = df_clustered['KMeans_Cluster'].value_counts().sort_index()
    for cl, count in size_series.items():
        pct = (count / len(df_clustered)) * 100
        print(f"Cluster {cl}: {count} siswa ({pct:.2f}%)")

    # Hubungan dengan nilai ujian akhir (Exam_Score)
    print("\nRata-rata Nilai Ujian Akhir (Exam_Score) per Cluster:")
    exam_by_cluster = df_clustered.groupby('KMeans_Cluster')['Exam_Score'].agg(['mean', 'median', 'std'])
    print(exam_by_cluster)

    print("\n" + "=" * 70)
    print("PROGRAM K-MEANS SELESAI DENGAN SUKSES!")
    print("=" * 70)

if __name__ == '__main__':
    run_kmeans()

