# 🫀 Sistema Predictivo DIALYSIS® — Estadificación CKM 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dialysis-ckm-2026.streamlit.app)

Sistema predictivo basado en inteligencia artificial y el **Método DIALYSIS®** para la clasificación automatizada del Síndrome Cardiovascular-Renal-Metabólico (CKM) en 4 estadios, conforme a la Guía AHA/ACC/ADA/ASN 2026.

---

## 📋 Descripción

El sistema clasifica el estadio CKM de cada paciente (0–3) a partir de 28 variables clínicas, metabólicas, antropométricas, conductuales y ambientales derivadas de la ENSANUT 2022.

| Estadio | Descripción |
|---------|-------------|
| **0** | Sin factores de riesgo CKM |
| **1** | Sobrepeso o prediabetes |
| **2** | Un criterio metabólico mayor |
| **3** | Múltiples criterios o daño orgánico |

---

## 🧬 Método DIALYSIS®

Metodología original de mejora continua y gestión de calidad diseñada para unidades de hemodiálisis.

**Registro INDAUTOR No. 03-2026-050609443300-01**  
Autora: Yuritzi Lorena Trujillo Vargas  
Institución: Centro Europeo de Másteres y Posgrados (CEMP)

| Componente | Descripción |
|-----------|-------------|
| **D** — Diagnóstico | Identificación del problema clínico |
| **I** — Indicadores | Selección de variables clave |
| **A** — Análisis | Interpretación estructurada de datos |
| **L** — Localización | Priorización mediante IA y SHAP |
| **Y** — Yield/Mejora | Ciclo PDSA y mejora continua |
| **S** — Sistema | SOPs digitales y trazabilidad |
| **I** — Inteligencia Artificial | Modelos predictivos (XGBoost) |
| **S** — Sostenibilidad | Cultura de mejora basada en datos |

---

## 🤖 Modelo

- **Algoritmo**: XGBoost (multiclase, 4 estadios CKM)
- **Dataset**: ENSANUT 2022 — 4,281 registros
- **Accuracy**: 97.8%
- **ROC-AUC**: 99.93% (OvR ponderado)
- **Validación**: Stratified K-Fold k=5 (media 98.6%, DE=0.005)
- **Interpretabilidad**: SHAP

---

## 🚀 Uso local

```bash
git clone https://github.com/dialysisdataconsulting-boop/dialysis-ckm-2026.git
cd dialysis-ckm-2026
pip install -r requirements.txt
streamlit run app.py
```

---

## 📚 Referencias

1. Ndumele CE, et al. 2026 AHA/ACC/ADA/ASN Guideline for CKM Syndrome. *Circulation*. 2026. doi:10.1161/CIR.0000000000001453
2. Trujillo Vargas YL. Método DIALYSIS®. Reg. INDAUTOR No. 03-2026-050609443300-01. México; 2026.
3. Instituto Nacional de Salud Pública. ENSANUT 2022. Cuernavaca: INSP; 2023.

---

## ⚕️ Aviso

Esta herramienta es de apoyo predictivo. No sustituye la valoración médica clínica.  
Los resultados deben ser interpretados por personal de salud calificado.

---

*DIALYSIS-DATA Consulting · dialysis.data.consulting@gmail.com · Ciudad de México, 2026*
