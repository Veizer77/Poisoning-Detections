# Poisoning-Detections


## 📁 Struktur Notebook

Berikut fungsi utama dari masing-masing file `.ipynb` dalam proyek:

### 1. **`demo.ipynb`**
- **Fungsi:** Menampilkan demonstrasi sistem secara interaktif.
- **Deskripsi:** Biasanya digunakan untuk presentasi, visualisasi hasil, atau simulasi sederhana.

### 2. **`evaluate_detector.ipynb`**
- **Fungsi:** Mengevaluasi performa model detektor terhadap data poisoning.
- **Deskripsi:** Mengukur metrik seperti *accuracy*, *precision*, *recall*, dan *F1-score*.

### 3. **`feature_extractor.ipynb`**
- **Fungsi:** Mengekstraksi fitur dari data mentah.
- **Deskripsi:** Mengubah data (teks/gambar/waktu) menjadi representasi numerik seperti *TF-IDF* atau *embedding*.

### 4. **`integration.ipynb`**
- **Fungsi:** Menggabungkan semua komponen sistem menjadi satu pipeline terpadu.
- **Deskripsi:** Mengintegrasikan detektor, ekstraktor fitur, dan sistem RAG untuk *end-to-end workflow*.

### 5. **`poison_detector.ipynb`**
- **Fungsi:** Melatih model untuk mendeteksi data beracun (*poisoned data*).
- **Deskripsi:** Mengimplementasikan algoritma deteksi dan mitigasi serangan terhadap model ML.

### 6. **`poison_simulator.ipynb`**
- **Fungsi:** Mensimulasikan serangan *poisoning* pada dataset.
- **Deskripsi:** Menghasilkan data sintetis yang diracuni untuk pengujian ketahanan model.

### 7. **`rag_system.ipynb`**
- **Fungsi:** Mengembangkan sistem **RAG (Retrieval-Augmented Generation)**.
- **Deskripsi:** Menggabungkan *retrieval* (pencarian) dan *generation* (pembuatan teks) untuk meningkatkan kemampuan model bahasa.

---

## 🔄 Alur Eksekusi Rekomendasi

### **FASE 1: PREPARASI DATA & SIMULASI**
```bash
1. poison_simulator.ipynb
   ↓
2. feature_extractor.ipynb
````

> Menghasilkan data racun dan mengekstraksi fitur siap latih.

---

### **FASE 2: DETEKSI & EVALUASI**

```bash
3. poison_detector.ipynb
   ↓
4. evaluate_detector.ipynb
```

> Melatih dan mengukur performa model detektor.

---

### **FASE 3: SISTEM RAG & INTEGRASI**

```bash
5. rag_system.ipynb
   ↓
6. integration.ipynb
```

> Membangun sistem RAG dan menggabungkannya dengan pipeline deteksi.

---

### **FASE 4: DEMONSTRASI**

```bash
7. demo.ipynb
```

> Menampilkan hasil akhir sistem yang sudah terintegrasi.

---

## 🧩 Panduan Eksekusi per Notebook

### `poison_simulator.ipynb`

* Generate *clean dataset*
* Tambahkan skenario serangan *poisoning*
* Simulasikan berbagai jenis serangan
* Simpan dataset hasil simulasi

### `feature_extractor.ipynb`

* Input: Dataset hasil simulasi
* Proses: Ekstraksi fitur
* Output: Vektor fitur untuk pelatihan model

### `poison_detector.ipynb`

* Input: Vektor fitur
* Proses: Latih model detektor
* Output: Model deteksi siap evaluasi

### `evaluate_detector.ipynb`

* Input: Model + data uji
* Proses: Evaluasi performa
* Output: Metrik seperti *accuracy*, *precision*, *recall*, *F1-score*

### `rag_system.ipynb`

* Bangun pipeline RAG
* Gunakan *vector database*, retrieval, dan *generator model*
* Hasil: Sistem RAG yang dapat menjawab atau menghasilkan informasi kontekstual

### `integration.ipynb`

* Integrasikan detektor + sistem RAG
* Ciptakan *end-to-end secure pipeline*

### `demo.ipynb`

* Jalankan simulasi dan demonstrasi sistem akhir
* Tampilkan kemampuan deteksi dan respons aman

---

## ⚙️ Prasyarat & Dependencies

Instal semua dependensi berikut sebelum menjalankan notebook:

```bash
pip install jupyter notebook
pip install pandas numpy scikit-learn
pip install torch transformers
pip install langchain chromadb
pip install matplotlib seaborn
```

