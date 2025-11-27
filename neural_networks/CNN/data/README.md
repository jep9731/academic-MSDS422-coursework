# 📥 Dataset Download Instructions

This project uses a Cats vs Dogs image dataset for training and evaluating Convolutional Neural Network (CNN) models.  
Because the dataset contains nearly 10,000 images, it is **not** stored directly in this repository.

Instead, use the provided script to automatically download and extract the dataset into the appropriate folder structure.

---

## 🔧 Requirements

Before running the script, ensure the following Python package is installed:

```bash
pip install gdown
```

`gdown` is required to download files from Google Drive.

---

## 🚀 Downloading the Dataset

From the root of the repository, run:

```bash
python download_data.py
```

This will:

* ✔ Download the dataset ZIP file from Google Drive
* ✔ Save it under data/dataset.zip
* ✔ Automatically extract all image files
* ✔ Create the following structure:
  ```bash
  data/
├─ train/
│   ├─ cats/
│   └─ dogs/
└─ test/
    ├─ cats/
    └─ dogs/
  ```

---

## 📌 Alternate Manual Download (Optional)

If you prefer to download manually, use the link below — then unzip the contents into a folder named `Images/`:

🔗 Dataset Link:
https://drive.google.com/file/d/1Xodq1tBD-udPhHA7x0sFXqPap6zyDVSz/view?usp=drive_link
