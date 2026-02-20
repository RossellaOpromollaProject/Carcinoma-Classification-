# Carcinoma-Classification
Progetto di Deep Learning per la classificazione multiclasse di immagini mediche tramite PyTorch.

Il modello predice tre classi:

- 0 → Not Detectable
- 1 → Benign
- 2 → Malignant

---

## Obiettivo

Sviluppare una pipeline completa per la classificazione di immagini mediche che includa:

- Caricamento e gestione del dataset
- Preprocessing e data augmentation
- Addestramento di una rete neurale convoluzionale (CNN)
- Valutazione tramite metriche di classificazione
- Matrice di confusione
- Supporto GPU (se disponibile)

---

## Dataset

Il dataset è strutturato tramite un file `metadata.csv` contenente:

- `id` → identificativo dell'immagine
- `malignant` → etichetta originale

Mappatura delle classi utilizzata nel modello:

```python
LABEL_MAP = {
    "not detectable": 0,
    "benign": 1,
    "malignant": 2
}
```

Le cartelle `dati` e `risultati` non sono incluse nel repository per motivi di dimensione.

Sono disponibili al seguente link:

https://drive.google.com/drive/folders/1WWsv58o20fEwElQHQvikMqaC4gU3IGB3?usp=drive_link

---

## Architettura del Modello

Il progetto utilizza una rete neurale convoluzionale (CNN) implementata in PyTorch per l'apprendimento supervisionato multiclasse.

Il training sfrutta GPU se disponibile.

---

## Tecnologie Utilizzate

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Esecuzione

1. Clonare il repository
2. Installare le dipendenze
3. Aprire il notebook `carcinoma_classification.ipynb`
4. Eseguire le celle in sequenza

---

## Finalità

Progetto sviluppato per finalità accademiche con l'obiettivo di applicare tecniche di Deep Learning alla classificazione di immagini mediche.
