import os
import sys
import tempfile
import warnings

# Configure environment variables to avoid matplotlib/loky issues on Windows
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder

def run_kmodes():
    print("=" * 70)
    print("PRAKTIKUM MACHINE LEARNING - CLUSTERING DENGAN K-MODES")
    print("Kelompok: GANJIL")
    print("Dataset: Student Performance Factors (Data Kategorikal)")
    print("=" * 70)

    # Direktori keluaran
    output_dir = os.path.join("outputs", "kmodes")
    os.makedirs(output_dir, exist_ok=True)

    # Style visualisasi
    try:
        plt.style.use('fivethirtyeight')
    except Exception:
        plt.style.use('default')

    # ==========================================
    # 1. MEMBACA DATASET
    # ==========================================
    dataset_path = os.path.join("data", "StudentPerformanceFactors.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {dataset_path}")

    print(f"\n[LANGKAH 1] Membaca dataset dari: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dimensi dataset asli: {df.shape[0]} baris, {df.shape[1]} kolom")

    # ==========================================
    # 2. PEMILIHAN FITUR KATEGORIKAL & PEMERIKSAAN KUALITAS DATA
    # ==========================================
    # Pemilihan fitur diprioritaskan pada faktor kebiasaan, dukungan lingkungan, dan motivasi siswa
    # sesuai dengan instruksi prompt.md:
    selected_categorical_features = [
        'Parental_Involvement', 
        'Access_to_Resources', 
        'Motivation_Level', 
        'Internet_Access', 
        'Family_Income', 
        'Teacher_Quality', 
        'Peer_Influence',
        'Extracurricular_Activities'
    ]

    print(f"\n[LANGKAH 2] Fitur kategorikal yang dipilih ({len(selected_categorical_features)} fitur):")
    for idx, col in enumerate(selected_categorical_features, 1):
        print(f" {idx}. {col}")

    # Pemeriksaan missing values pada fitur kategorikal yang dipilih
    df_cat = df[selected_categorical_features].copy()
    missing_cat = df_cat.isnull().sum()
    print("\nPemeriksaan Missing Values:")
    print(missing_cat)

    # Penanganan Missing Values:
    # 'Teacher_Quality' memiliki 78 nilai null (~1.18% dari data).
    # Dilakukan imputasi nilai modus (kategori paling sering muncul) untuk mempertahankan integritas data kategorikal.
    for col in df_cat.columns:
        if df_cat[col].isnull().sum() > 0:
            mode_val = df_cat[col].mode()[0]
            print(f"Imputasi kolom '{col}' dengan nilai modus: '{mode_val}'")
            df_cat[col] = df_cat[col].fillna(mode_val)

    # Pemeriksaan kategori unik dan frekuensi
    print("\nRingkasan Kategori Unik:")
    for col in df_cat.columns:
        counts = df_cat[col].value_counts().to_dict()
        print(f" - {col}: {counts}")

    # ==========================================
    # 3. PENGUJIAN BEBERAPA NILAI K (ELBOW DENGAN COST K-MODES)
    # ==========================================
    # Cost pada K-Modes adalah total dissimilarity (jumlah ketidakcocokan kategori/Hamming distance)
    # antara setiap observasi dengan centroid modusnya.
    print("\n[LANGKAH 3] Menguji beberapa nilai k (k=2..7) dengan K-Modes (init='Huang')...")
    k_range = list(range(2, 8))
    costs = []
    avg_dissimilarities = []
    hamming_silhouettes = []

    # Encode ke ordinal hanya untuk keperluan perhitungan Hamming distance Silhouette Score
    encoder = OrdinalEncoder()
    df_cat_encoded = encoder.fit_transform(df_cat)

    for k in k_range:
        km_test = KModes(n_clusters=k, init='Huang', n_init=10, random_state=42, verbose=0)
        km_labels = km_test.fit_predict(df_cat)
        c = km_test.cost_
        avg_dissim = c / len(df_cat)
        costs.append(c)
        avg_dissimilarities.append(avg_dissim)
        
        # Metrik evaluasi tambahan: Silhouette Score berbasis Hamming distance
        sil_h = silhouette_score(df_cat_encoded, km_labels, metric='hamming')
        hamming_silhouettes.append(sil_h)
        print(f"k = {k} | Cost = {c:8.1f} | Avg Dissimilarity/Siswa = {avg_dissim:.3f} | Hamming Silhouette = {sil_h:.4f}")

    # Simpan hasil perbandingan evaluasi ke file CSV
    eval_df = pd.DataFrame({
        'k': k_range,
        'Cost': costs,
        'Avg_Dissimilarity_Per_Student': avg_dissimilarities,
        'Hamming_Silhouette_Score': hamming_silhouettes
    })
    eval_csv_path = os.path.join(output_dir, "kmodes_evaluation_metrics.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"\nEvaluasi metrik k-modes tersimpan di: {eval_csv_path}")

    # Visualisasi Kurva Cost Elbow K-Modes
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, costs, 'o-', color='darkred', markersize=8, linewidth=2)
    plt.title('Cost vs Jumlah Cluster pada K-Modes (Elbow Method)', fontsize=14)
    plt.xlabel('Jumlah Cluster (k)', fontsize=12)
    plt.ylabel('Cost (Total Ketidakcocokan Kategori)', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.5)

    elbow_path = os.path.join(output_dir, "01_kmodes_cost_elbow.png")
    plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar kurva cost tersimpan di: {elbow_path}")

    # ==========================================
    # 4. MEMBANGUN MODEL K-MODES FINAL
    # ==========================================
    # Pemilihan k optimal:
    # Memilih k = 3. Cost terus menurun seiring bertambahnya k karena overfitting wajar pada partisi,
    # namun k = 3 memberikan pemisahan profil sosio-akademis yang paling interpretable:
    # - Kelompok Dukungan Tinggi (High Support & High Motivation)
    # - Kelompok Dukungan Menengah/Mandiri (Moderate Support & Low/Medium Motivation)
    # - Kelompok Dukungan Terbatas/Rentan (Low Support/Resources)
    optimal_k = 3
    print(f"\n[LANGKAH 4] Membangun model K-Modes final dengan k = {optimal_k}...")
    km_final = KModes(n_clusters=optimal_k, init='Huang', n_init=10, random_state=42, verbose=0)
    final_labels = km_final.fit_predict(df_cat)

    # Tambahkan label ke salinan DataFrame (tidak mengubah file asli)
    df_clustered = df.copy()
    df_clustered['KModes_Cluster'] = final_labels

    # Tampilkan modus/centroid masing-masing cluster
    centroids_df = pd.DataFrame(km_final.cluster_centroids_, columns=selected_categorical_features)
    centroids_df.index = [f"Cluster {i}" for i in range(optimal_k)]
    print("\nModus Kategori pada Tiap Cluster (Centroid K-Modes):")
    print(centroids_df.to_string())

    centroids_csv_path = os.path.join(output_dir, "kmodes_centroids_modes.csv")
    centroids_df.to_csv(centroids_csv_path)
    print(f"Modus centroid tersimpan di: {centroids_csv_path}")

    # ==========================================
    # 5. DISTRIBUSI UKURAN & PROFIL CLUSTER
    # ==========================================
    print("\n[LANGKAH 5] Ukuran Masing-Masing Cluster:")
    cluster_counts = df_clustered['KModes_Cluster'].value_counts().sort_index()
    for cl_id, count in cluster_counts.items():
        pct = (count / len(df_clustered)) * 100
        print(f"Cluster {cl_id}: {count} siswa ({pct:.2f}%)")

    # Visualisasi ukuran cluster
    plt.figure(figsize=(8, 5))
    colors = ['#3498db', '#e67e22', '#2ecc71']
    bars = plt.bar([f"Cluster {i}" for i in range(optimal_k)], cluster_counts.values, color=colors, width=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{yval} ({yval/len(df_clustered)*100:.1f}%)", ha='center', va='bottom', fontsize=10)
    plt.title('Distribusi Ukuran Siswa per Cluster K-Modes', fontsize=13)
    plt.ylabel('Jumlah Siswa', fontsize=11)
    plt.ylim(0, max(cluster_counts.values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    size_plot_path = os.path.join(output_dir, "02_kmodes_cluster_distribution.png")
    plt.savefig(size_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar distribusi ukuran cluster tersimpan di: {size_plot_path}")

    # ==========================================
    # 6. VISUALISASI KOMPOSISI FITUR KATEGORIKAL PER CLUSTER
    # ==========================================
    print("\n[LANGKAH 6] Membuat Visualisasi Komposisi Fitur per Cluster...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    features_to_breakdown = ['Parental_Involvement', 'Motivation_Level', 'Access_to_Resources', 'Family_Income']
    
    for idx, feat in enumerate(features_to_breakdown):
        ax = axes[idx // 2, idx % 2]
        # Hitung persentase fitur per cluster
        prop_df = pd.crosstab(df_clustered['KModes_Cluster'], df_clustered[feat], normalize='index') * 100
        prop_df.plot(kind='bar', ax=ax, colormap='tab10', width=0.7)
        ax.set_title(f'Distribusi Persentase: {feat}', fontsize=12)
        ax.set_xlabel('Cluster K-Modes', fontsize=10)
        ax.set_ylabel('Persentase Siswa (%)', fontsize=10)
        ax.set_xticklabels([f"Cluster {i}" for i in range(optimal_k)], rotation=0)
        ax.legend(title=feat, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, "03_kmodes_feature_composition.png")
    plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar komposisi fitur tersimpan di: {comp_plot_path}")

    # ==========================================
    # 7. PERBANDINGAN PERFORMA UJIAN PADA CLUSTER KATEGORIKAL (POST-HOC)
    # ==========================================
    print("\n[LANGKAH 7] Evaluasi Performa Ujian (Exam_Score) pada Cluster Kategorikal:")
    exam_stats = df_clustered.groupby('KModes_Cluster')['Exam_Score'].agg(['count', 'mean', 'median', 'std'])
    print(exam_stats)

    # Boxplot Exam_Score per cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='KModes_Cluster', y='Exam_Score', hue='KModes_Cluster', data=df_clustered, palette='Set2', legend=False)
    plt.title('Distribusi Nilai Ujian (Exam_Score) Berdasarkan Cluster K-Modes', fontsize=13)
    plt.xlabel('Cluster K-Modes (Lingkungan & Karakteristik Sosio-Akademis)', fontsize=11)
    plt.ylabel('Nilai Ujian (Exam Score)', fontsize=11)
    plt.xticks([0, 1, 2], [f"Cluster {i}\n(n={cluster_counts[i]})" for i in range(optimal_k)])
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    exam_plot_path = os.path.join(output_dir, "04_kmodes_exam_score_comparison.png")
    plt.savefig(exam_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gambar perbandingan Exam_Score tersimpan di: {exam_plot_path}")

    # ==========================================
    # 8. MENYIMPAN RINGKASAN PROFIL KATEGORIKAL
    # ==========================================
    # Buat tabel frekuensi kategori per cluster untuk semua fitur yang dianalisis
    profile_summary_list = []
    for feat in selected_categorical_features:
        ct = pd.crosstab(df_clustered['KModes_Cluster'], df_clustered[feat], normalize='index') * 100
        ct['Fitur'] = feat
        profile_summary_list.append(ct)
    
    full_profile_df = pd.concat(profile_summary_list)
    profile_csv_path = os.path.join(output_dir, "kmodes_category_breakdown.csv")
    full_profile_df.to_csv(profile_csv_path)
    print(f"Tabel persentase kategori per cluster tersimpan di: {profile_csv_path}")

    print("\n" + "=" * 70)
    print("PROGRAM K-MODES SELESAI DENGAN SUKSES!")
    print("=" * 70)

if __name__ == '__main__':
    run_kmodes()
