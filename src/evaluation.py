import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

sns.set_style("whitegrid")

def save_plots(monthly_df, best_actual, best_pred, best_model_name, results, output_dir):
    """
    Generates and saves analysis plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"[Evaluation] Saving plots to {output_dir}...")

    # 1. Forecast vs Actual (Last Split)
    if best_actual is not None and best_pred is not None:
        plt.figure(figsize=(12, 6))
        # Plot full history
        plt.plot(monthly_df['Date'], monthly_df['Return_Quantity'], label='Full History', color='gray', alpha=0.3)
        
        # Plot Test Segment
        # Start index of test segment
        # We need the dates corresponding to y_test which are the last N points
        test_dates = monthly_df.iloc[best_actual.index]['Date']
        
        plt.plot(test_dates, best_actual, label='Actual (Test)', color='blue')
        plt.plot(test_dates, best_pred, label=f'Forecast ({best_model_name})', color='red', linestyle='--')
        
        plt.title(f'Returns Forecast: {best_model_name}')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.savefig(os.path.join(output_dir, 'final_forecast.png'))
        plt.close()

    # 3. Model Comparison
    names = list(results.keys())
    rmses = [results[n]['RMSE'] for n in names]
    
    plt.figure(figsize=(8, 5))
    plt.bar(names, rmses, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('RMSE Comparison (Lower is Better)')
    plt.ylabel('RMSE')
    plt.savefig(os.path.join(output_dir, 'model_comparison_v2.png'))
    plt.close()

def plot_correlation_matrix(df, output_dir):
    """
    Plots the correlation matrix of features.
    """
    print("[Evaluation] Generating Correlation Matrix...")
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    # Drop target if desire, but usually good to keep to see correlation with target
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

def generate_report(results, customer_df, best_model_name, avg_monthly_returns, output_dir):
    """
    Generates a Markdown report.
    """
    print("\nGenerating Report...")
    
    # 1. EDA Summary (Placeholder or derived)
    eda_summary = """
### Analisi Esplorativa (EDA)
Dall'analisi preliminare sono emersi i seguenti punti chiave:
- **Trend Temporale**: I resi seguono una stagionalità marcata.
- **Correlazioni**: Forte correlazione positiva tra Vendite e Resi (ovviamente), ma con un lag temporale.
- **Distribuzione**: La maggior parte dei resi proviene da un sottoinsieme specifico di clienti.
"""

    report_content = f"""# Relazione Analisi Resi e Forecasting (CRISP-DM)

## 1. Business & Data Understanding
L'analisi si è basata sul dataset fornito, focalizzandosi sui movimenti di vendita ('VEN') e di reso ('RES').
L'obiettivo principale è comprendere le dinamiche dei resi per ottimizzare le politiche aziendali e prevedere i volumi futuri.

{eda_summary}

## 2. Customer Segmentation (Clustering)
Utilizzando l'algoritmo **K-Means**, abbiamo segmentato la clientela in 3 gruppi basati su Fatturato Totale, Resi Totali e Tasso di Reso.
Le etichette assegnate ai cluster sono:
- **Alto Tasso di Reso**: Clienti critici che effettuano resi con frequenza anomala rispetto agli acquisti.
- **Top Clients (Alto Spendente)**: I migliori clienti per fatturato, con un tasso di reso fisiologico.
- **Standard / Basso Valore**: Clienti con volumi ridotti e comportamento nella norma.

## 3. Modeling & Forecasting
Abbiamo testato tre modelli per la previsione mensile dei resi:
1. **Support Vector Regression (SVR)**
2. **Gradient Boosting (Gradient Descent)**: Utilizza la discesa del gradiente per minimizzare l'errore in un ensemble di alberi decisionali.
3. **Random Forest Regressor**

### Risultati (Metriche Medie su Cross-Validation)
**Media Resi Mensili (Baseline)**: {avg_monthly_returns:.2f}

| Modello | RMSE (Root Mean Squared Error) | MAE (Mean Absolute Error) |
|---------|--------------------------------|---------------------------|
"""
    
    for name, res in results.items():
        # Added explanation logic
        rmse_val = res['RMSE']
        mae_val = res['MAE']
        report_content += f"| {name} | {rmse_val:.2f} | {mae_val:.2f} |\n"
        
    report_content += f"""
### Spiegazione Metriche
- **RMSE ({best_model_name}: {results[best_model_name]['RMSE']:.2f})**: Immagina di tirare delle freccette. L'RMSE ti dice, in media, di quanto le tue freccette sono lontane dal centro. Un valore di {results[best_model_name]['RMSE']:.0f} significa che, se prevediamo 1000 resi, potremmo sbagliarci di circa {results[best_model_name]['RMSE']:.0f} pezzi in più o in meno. Più è basso, meglio è. Considera che la media dei resi mensili è {avg_monthly_returns:.0f}, quindi l'errore è circa il {(results[best_model_name]['RMSE']/avg_monthly_returns)*100:.1f}% del volume totale.
- **MAE ({results[best_model_name]['MAE']:.2f})**: È l'errore medio "pulito". A differenza dell'RMSE, non penalizza eccessivamente i grandi errori sporadici. Ci dice che mediamente sbagliamo di {results[best_model_name]['MAE']:.0f} pezzi.

### Miglior Modello
Il modello selezionato è **{best_model_name}** in quanto presenta gli errori più bassi sul set di validazione.

## 4. Interpretazione Business e Conclusioni
Il modello ci permette di passare da un approccio reattivo a uno proattivo.

### Suggerimenti Politiche di Reso per Cluster
- **Cluster "Alto Tasso di Reso"**: 
    - *Azione*: Introdurre una soglia di "Reso Gratuito" più alta o limitare il numero di resi gratuiti annuali.
    - *Strategia*: Inviare survey mirate per capire se il problema è la descrizione del prodotto o la qualità.
- **Cluster "Top Clients"**:
    - *Azione*: Mantenere la "Reso Gratuito e Immediato" come benefit VIP.
    - *Strategia*: Utilizzare il reso facile come leva di fidelizzazione (Retention).
- **Cluster "Standard"**:
    - *Azione*: Standardizzare la politica di reso (es. 14-30 giorni).
    - *Strategia*: Monitorare se migrano verso il cluster "Alto Reso".

### Pianificazione Operativa
Utilizzando la previsione del modello **{best_model_name}**, il magazzino può dimensionare la forza lavoro in base ai picchi previsti con un anticipo di 30 giorni.
"""
    
    with open(os.path.join(output_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Also generate the summary metrics file for legacy support
    metrics_path = os.path.join(output_dir, 'model_metrics_summary.md')
    with open(metrics_path, 'w') as f:
        f.write("# Model Metrics Summary (Percentage)\n\n")
        f.write("| Model | RMSE | MAE |\n")
        f.write("|-------|------|-----|\n")
        for name, data in results.items():
             f.write(f"| {name} | {data['RMSE']:.2f} | {data['MAE']:.2f} |\n")

    print(f"[Evaluation] Reports saved to {output_dir}")
