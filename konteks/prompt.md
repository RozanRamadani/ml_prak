# TUGAS PRAKTIKUM CLUSTERING — KELOMPOK GANJIL

## 1. Peran dan tujuan

Bertindaklah sebagai asisten data science dan machine learning yang membantu menyelesaikan tugas praktikum clustering menggunakan Python dan scikit-learn.

Kerjakan proyek secara menyeluruh berdasarkan modul `MODUL 13 Clustering.pdf` yang tersedia di workspace. Saya termasuk **kelompok ganjil**, sehingga tugas algoritma clustering tambahan yang wajib dikerjakan adalah **K-Modes**, bukan K-Prototype.

Jangan hanya memberikan tutorial atau potongan kode. Periksa workspace, buat file yang diperlukan, implementasikan program, jalankan pengujian, analisis hasil aktual, dan susun laporan berdasarkan hasil eksekusi program.

## 2. Dataset

Gunakan dataset berikut sebagai satu-satunya dataset utama:

URL: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

Nama dataset: Student Performance Factors.

Instruksi:
- Periksa terlebih dahulu apakah file CSV sudah tersedia di workspace.
- Jika belum tersedia, cari file CSV yang telah diunduh dari Kaggle.
- Jika file belum ada dan internet atau akses Kaggle tersedia, unduh dataset dari sumber resmi tersebut.
- Jika tidak bisa mengunduh dataset, jangan membuat dataset sintetis atau mengarang hasil. Buat program yang siap dijalankan dan jelaskan bahwa dataset perlu diletakkan di lokasi yang ditentukan.
- Pertahankan file dataset asli tanpa mengubah atau menimpanya.
- Pastikan nama kolom dan tipe data dibaca dari CSV aktual, bukan berdasarkan asumsi.

## 3. Pelajari modul dosen

Baca `MODUL 13 Clustering.pdf` terlebih dahulu dan ikuti struktur praktikumnya.

Modul mencakup:
1. Import library.
2. Membaca dataset.
3. Memeriksa bentuk dataset menggunakan `shape`.
4. Statistik deskriptif.
5. Pemeriksaan missing values.
6. Pemeriksaan outlier.
7. Visualisasi distribusi masing-masing fitur.
8. Visualisasi hubungan antarf fitur.
9. Visualisasi sebaran berdasarkan kategori jika tersedia.
10. Menentukan jumlah cluster dengan Elbow Method.
11. Membangun model K-Means.
12. Menyiapkan data untuk visualisasi cluster.
13. Memvisualisasikan hasil clustering.
14. Menghitung Silhouette Score.

Lengkapi bagian kode yang masih bertanda titik-titik pada modul dengan menyesuaikannya terhadap dataset Student Performance Factors. Jangan menyalin nama kolom dari dataset Mall Customers apabila kolom tersebut tidak tersedia.

## 4. Program pertama: K-Means

Buat file `kmeans_clustering.py`.

Persyaratan:
- Gunakan Python, pandas, NumPy, matplotlib, seaborn, dan scikit-learn.
- Periksa dimensi data, tipe data, statistik deskriptif, missing values, duplikasi, dan outlier.
- Visualisasikan distribusi fitur yang relevan.
- Pilih fitur numerik yang sesuai untuk clustering dan jelaskan alasan pemilihannya.
- Jangan menggunakan `Exam_Score` sebagai fitur clustering jika tujuan analisisnya adalah menemukan kelompok berdasarkan faktor yang memengaruhi performa siswa. Jika nilai ujian tetap digunakan, jelaskan bahwa tujuan analisis berubah dan interpretasikan hasil secara hati-hati.
- Tangani missing values menggunakan metode yang sesuai.
- Lakukan standardisasi fitur numerik sebelum K-Means jika skala fitur berbeda.
- Gunakan Elbow Method untuk mengevaluasi beberapa nilai k, misalnya 1 sampai 10, atau rentang yang sesuai dengan ukuran data.
- Buat grafik Elbow.
- Pilih jumlah cluster berdasarkan grafik dan alasan yang dapat dipertanggungjawabkan; jangan mengarang hasil elbow.
- Bangun model `sklearn.cluster.KMeans` dengan `random_state` tetap dan `n_init` yang eksplisit.
- Tambahkan label cluster ke salinan DataFrame, bukan mengubah file CSV asli.
- Visualisasikan hasil cluster dengan scatter plot yang sesuai.
- Jika fitur lebih dari dua, gunakan dua fitur yang relevan atau PCA hanya untuk visualisasi. Jangan mengganti fitur model secara diam-diam.
- Hitung Silhouette Score hanya ketika jumlah cluster dan distribusi label memenuhi persyaratan metrik.
- Simpan grafik dan ringkasan hasil ke folder `outputs/`.
- Cetak ukuran setiap cluster dan profil rata-rata atau karakteristik fitur per cluster.
- Berikan interpretasi yang mudah dipahami tentang perbedaan karakteristik antarcluster.

## 5. Program kedua: K-Modes untuk kelompok ganjil

Buat file `kmodes_clustering.py`.

Gunakan algoritma K-Modes yang sesuai untuk data kategorikal. Jika library `kmodes` belum terpasang, cantumkan dependensi `kmodes` dalam `requirements.txt` dan berikan instruksi instalasi.

Persyaratan:
- Pilih fitur kategorikal yang benar-benar tersedia di dataset.
- Jangan mengubah variabel kategorikal menjadi angka integer biasa lalu menganggap selisih angka sebagai jarak yang bermakna.
- Periksa missing values, kategori langka, dan kategori yang tidak konsisten.
- Tangani missing values secara eksplisit.
- Gunakan `kmodes.kmodes.KModes` dengan `init='Huang'` atau metode inisialisasi yang sesuai, `n_init` eksplisit, dan `random_state` tetap jika didukung oleh versi library.
- Uji beberapa nilai k yang masuk akal dan bandingkan cost K-Modes. Jelaskan bahwa cost yang lebih kecil tidak otomatis berarti jumlah cluster terbaik.
- Jika menghitung metrik evaluasi tambahan, gunakan metrik yang sesuai untuk clustering kategorikal dan jelaskan keterbatasannya.
- Tambahkan label cluster pada salinan DataFrame.
- Tampilkan ukuran setiap cluster, modus kategori pada tiap cluster, dan tabel profil cluster.
- Visualisasikan distribusi kategori atau komposisi cluster menggunakan grafik yang sesuai.
- Jangan mengarang hasil clustering. Semua angka, tabel, dan interpretasi harus berasal dari eksekusi aktual.
- Simpan grafik dan hasil ringkasan ke folder `outputs/`.

Prioritaskan fitur kategorikal yang relevan dengan kebiasaan dan kondisi siswa, misalnya tingkat keterlibatan orang tua, tingkat motivasi, akses internet, atau fitur kategorikal lain yang benar-benar tersedia. Jangan menggunakan fitur yang tidak ditemukan pada dataset.

## 6. Perbandingan K-Means dan K-Modes

Buat analisis perbandingan yang mencakup:
- Jenis data yang digunakan.
- Cara algoritma mengelompokkan data.
- Fitur yang dipilih.
- Metode menentukan jumlah cluster.
- Hasil ukuran cluster.
- Karakteristik utama setiap cluster.
- Metrik evaluasi yang digunakan dan batasannya.
- Kelebihan serta kekurangan masing-masing algoritma untuk dataset ini.

Jangan membandingkan angka evaluasi secara langsung jika metrik atau jenis representasi datanya tidak sebanding.

## 7. Struktur file proyek

Buat struktur seperti berikut:

student-performance-clustering/
├── data/
│   └── StudentPerformanceFactors.csv
├── outputs/
│   ├── kmeans/
│   └── kmodes/
├── kmeans_clustering.py
├── kmodes_clustering.py
├── requirements.txt
├── README.md
└── laporan_praktikum.md

Jika dataset tidak tersedia, jangan membuat file CSV palsu. Jelaskan cara menempatkan dataset pada folder `data/`.

Sesuaikan nama file dataset dengan file aktual yang ditemukan dan pastikan kedua program menggunakan lokasi yang konsisten.

## 8. Laporan praktikum

Buat laporan `laporan_praktikum.md` berdasarkan hasil eksekusi program yang sebenarnya, dengan struktur:

1. Judul praktikum dan identitas kelompok.
2. Tujuan praktikum.
3. Deskripsi dataset dan sumber Kaggle.
4. Landasan teori singkat tentang clustering, K-Means, dan K-Modes.
5. Persiapan data dan preprocessing.
6. Implementasi K-Means.
7. Hasil Elbow Method dan alasan pemilihan k.
8. Visualisasi serta interpretasi cluster K-Means.
9. Implementasi K-Modes.
10. Hasil pemilihan jumlah cluster K-Modes.
11. Visualisasi serta interpretasi cluster K-Modes.
12. Perbandingan kedua algoritma.
13. Kesimpulan.
14. Referensi.

Masukkan gambar hasil aktual dari folder `outputs/` ke laporan. Jika laporan menggunakan Markdown, gunakan path gambar relatif agar mudah dibuka.

Jangan menulis angka hasil, Silhouette Score, jumlah anggota cluster, atau kesimpulan empiris sebelum program benar-benar dijalankan.

## 9. README dan dependensi

Buat `README.md` yang menjelaskan:
- Tujuan proyek.
- Struktur folder.
- Cara memasang dependensi.
- Cara menempatkan dataset.
- Cara menjalankan masing-masing program.
- Lokasi hasil visualisasi.
- Cara membaca hasil clustering.

Buat `requirements.txt` dengan dependensi yang diperlukan dan versi yang kompatibel bila memungkinkan. Jangan memasukkan library yang tidak digunakan.

## 10. Pengujian dan validasi

Setelah implementasi:
1. Periksa syntax semua file Python.
2. Instal dependensi jika lingkungan mengizinkan.
3. Jalankan kedua program menggunakan dataset aktual.
4. Perbaiki error yang ditemukan.
5. Pastikan semua grafik dan tabel keluaran tersimpan.
6. Pastikan jumlah label cluster sama dengan jumlah baris yang digunakan untuk clustering.
7. Pastikan tidak ada kebocoran target atau kolom identitas yang tidak relevan.
8. Pastikan hasil dapat direproduksi menggunakan random seed yang tetap.
9. Jika pengujian tidak dapat dijalankan, jelaskan penyebabnya secara spesifik dan jangan mengklaim program berhasil.

## 11. Batasan penting

- Jangan mengubah dataset asli.
- Jangan mengarang isi dataset, metrik, hasil, screenshot, atau kesimpulan.
- Jangan melakukan penghapusan baris secara sembarangan tanpa menjelaskan alasannya.
- Jangan mengklaim K-Modes lebih baik hanya karena cost lebih kecil.
- Jangan menganggap cluster sebagai kategori siswa yang secara otomatis benar secara akademis; cluster hanya menunjukkan pola berdasarkan fitur yang dipilih.
- Hindari penggunaan `Exam_Score` sebagai fitur jika tujuan utamanya menganalisis faktor siswa, lalu membandingkan skor ujian sebagai variabel interpretasi setelah clustering.
- Jelaskan seluruh keputusan preprocessing dan pemilihan fitur.

## 12. Hasil akhir yang harus diserahkan

Selesaikan proyek sampai file-file yang dibutuhkan tersedia dan dapat diperiksa. Pada akhir pengerjaan, berikan ringkasan:
- File yang dibuat.
- Dataset yang digunakan dan jumlah baris/kolom aktual.
- Fitur yang digunakan oleh K-Means dan K-Modes.
- Nilai k yang dipilih dan alasannya berdasarkan hasil aktual.
- Metrik evaluasi aktual.
- Lokasi grafik dan laporan.
- Status pengujian.
- Hal yang masih perlu dilakukan secara manual.

Fokus pada ketepatan akademik, kode yang dapat dijalankan, interpretasi yang masuk akal, dan kesesuaian dengan MODUL 13 Clustering serta ketentuan kelompok ganjil.