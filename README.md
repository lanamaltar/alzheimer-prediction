# Sustav za predikciju Alzheimerove bolesti  

Ovaj repozitorij sadrži tri glavne komponente:  
1. **Model 1 (Logistička regresija)** – koristi demografske i kliničke podatke.  
2. **Model 2 (ResNet-50 CNN)** – koristi MR slike mozga.  
3. **Web aplikacija (Flask + Angular)** – integracija oba modela kroz jednostavno korisničko sučelje.  

---

## Model 1 – Logistička regresija  

Ovaj projekt koristi **OASIS longitudinal** skup podataka (`oasis_longitudinal.csv`) za binarnu klasifikaciju stanja **Demented** vs **Nondemented**.  

### Kako pokrenuti  
1. Postaviti `oasis_longitudinal.csv` u istu mapu kao i notebook.  
2. Otvoriti `model1.ipynb` u Jupyteru ili koristiti VS Code (Jupyter ekstenzija).  
3. Pokrenuti ćelije redom (ili **Run All**).  

### Podaci  
Dataset korišten za treniranje i evaluaciju modela:  
[OASIS longitudinal MRI dataset](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers)  

---

## Model 2 – Duboko učenje (ResNet-50)  

Ovaj projekt koristi **MRI and Alzheimer’s dataset** za klasifikaciju više stadija demencije na temelju MRI slika mozga. Glavni kod je u notebooku `model2.ipynb`.  

### Kako pokrenuti  
1. Preuzeti dataset s poveznice ispod i raspakirati ga u projektnu mapu.  
2. Otvoriti `model2.ipynb` u Jupyteru ili koristiti VS Code (Jupyter ekstenzija).  
3. Pokrenuti ćelije redom (ili **Run All**).  

### Podaci  
Dataset korišten za treniranje i evaluaciju modela:  
[MRI Alzheimer’s dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)  

---

## Web aplikacija (Flask + Angular)  

Aplikacija integrira oba modela (Model 1 i Model 2) u obliku web aplikacije s **Flask backendom** i **Angular frontendom**.  

### Pokretanje servera (Flask backend)  
cd server
pip install -r requirements.txt
python app.py

### Pokretanje klijenta (Angular frontend)  
cd client
npm install
ng serve --proxy-config proxy.conf.json

