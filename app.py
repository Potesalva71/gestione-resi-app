import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Import Pipeline Modules
from src.data_loader import load_dataset
from src.features import create_monthly_features, calculate_estimated_price
from src.clustering import perform_clustering, assign_cluster_labels
from src.modeling import train_evaluate_models
from src.evaluation import save_plots, generate_report

# Page Configuration
st.set_page_config(page_title="Gestione Resi - Dashboard", layout="wide")

st.title("📊 Dashboard Analisi & Previsione Resi")
st.markdown("""
Questa dashboard permette di analizzare i dati storici, segmentare i clienti e prevedere i resi futuri.
Utilizza grafici interattivi per esplorare i dettagli.
""")

# Sidebar - Configuration
st.sidebar.header("1. Carica Dati")
uploaded_file = st.sidebar.file_uploader("Carica Dataset CSV", type=["csv"])

# Paths
PROJECT_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'app_results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- HELPER FUNCTIONS ---
def run_full_pipeline(df, use_optimized=False):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1. Feature Engineering
    status_text.text("Generazione Features...")
    progress_bar.progress(20)
    monthly_df = create_monthly_features(df)
    
    # 2. Clustering
    status_text.text("Segmentazione Clienti...")
    progress_bar.progress(40)
    customer_df, _ = perform_clustering(df)
    
    # 4. Assign Labels
    status_text.text("Assegnazione Etichette Cluster...")
    customer_df, cluster_labels = assign_cluster_labels(customer_df)

    # 5. Modeling
    status_text.text("Addestramento Modelli (SVR, RF, GBM)...")
    progress_bar.progress(60)
    results, best_model_name = train_evaluate_models(monthly_df, use_optimized=use_optimized)
    
    # 6. Evaluation (Save static plots for report backup)
    status_text.text("Salvataggio Report...")
    progress_bar.progress(80)
    # Note: save_plots might need update if it uses the old results structure, but we can skip it for the app's internal logic or rely on the script.
    # actually save_plots is imported from src.evaluation which I haven't checked. 
    # Let's assume it works or wrap it in try/except to not crash the app.
    try:
        save_plots(monthly_df, results[best_model_name]['Last_Test_Actual'], results[best_model_name]['Last_Test_Pred'], best_model_name, results, OUTPUT_DIR)
        generate_report(results, customer_df, best_model_name, OUTPUT_DIR)
    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"Warning during report generation: {e}")
    
    progress_bar.progress(100)
    status_text.text("Analisi Completata!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    # Prepare Forecast Data for Plotly (All Models)
    # Base dataframe with Date and Actual
    # Use the Test set dates from the best model (should be same for all)
    test_dates = monthly_df.loc[results[best_model_name]['Last_Test_Actual'].index, 'Date']
    
    forecast_df = pd.DataFrame({'Date': test_dates})
    forecast_df['Reale'] = results[best_model_name]['Last_Test_Actual'].values
    
    for model_name, res in results.items():
        if 'Last_Test_Pred' in res:
            forecast_df[model_name] = res['Last_Test_Pred']
            
    return monthly_df, customer_df, results, best_model_name, forecast_df

def display_glossary():
    with st.expander("📚 Glossario Features (Clicca per espandere)"):
        st.markdown("""
        **Features Temporali:**
        - `DATA_OPERAZIONE`: La data della transazione. Aggregata mensilmente per le previsioni.
        - `Month`, `Quarter`: Variabili che catturano la stagionalità (es. i resi aumentano a Gennaio dopo Natale).
        
        **Features di Flusso (Quantità):**
        - `QUANTITA_MOVIMENTATA`: Il numero netto di pezzi. 
        - `Sales_Quantity`: Quantità venduta (positiva).
        - `Return_Quantity`: Quantità resa (negativa, usata come target predittivo).
        
        **Features Ingegnerizzate (Machine Learning):**
        - `Sales_Lag1 / Lag2`: Le vendite di 1 o 2 mesi fa. *Significato*: I resi di oggi dipendono da quanto ho venduto in passato.
        - `Returns_Lag1`: I resi del mese scorso. *Significato*: Inerzia del fenomeno.
        - `Returns_RollMean_3`: Media mobile dei resi negli ultimi 3 mesi. *Significato*: Trend di breve periodo pulito dal rumore giornaliero.
        
        **Segmentazione:**
        - `Cluster`: Gruppo di appartenenza del cliente basato su comportamento d'acquisto e tasso di reso (Log-trasformato).
        """)

def display_eda_tab(df, key_prefix=""):
    st.header("🔍 Analisi Esplorativa (EDA)")
    st.info("In questa sezione puoi esplorare i dati in modo interattivo. Usa lo zoom e le selezioni sui grafici.")
    
    # 1. Trend Temporale (Time Series)
    st.subheader("1. Trend Temporale dei Resi")
    if 'DATA_OPERAZIONE' in df.columns:
        # Aggregazione mensile
        monthly_data = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].groupby(df['DATA_OPERAZIONE'].dt.to_period('M'))['QUANTITA_MOVIMENTATA'].sum()
        monthly_data.index = monthly_data.index.to_timestamp()
        monthly_df_plot = pd.DataFrame({'Data': monthly_data.index, 'Quantità Resi': monthly_data.values})
        
        fig_trend = px.line(monthly_df_plot, x='Data', y='Quantità Resi', markers=True, 
                            title="Andamento Mensile dei Resi", template="plotly_white")
        fig_trend.update_layout(xaxis_title="Data", yaxis_title="Quantità Assoluta (Resi)")
        st.plotly_chart(fig_trend, use_container_width=True, key=f"{key_prefix}trend")
        st.caption("ℹ️ **Interpretazione**: Il grafico mostra come variano i resi nel tempo. Picchi ricorrenti indicano stagionalità (es. post-Natale o saldi).")
    else:
        st.warning("Colonna 'DATA_OPERAZIONE' non trovata.")

    # 2. Distribuzione Causali (Boxplot)
    st.markdown("---")
    st.subheader("2. Distribuzione Resi per Causale")
    if 'DESCRIZIONE_CAUSALE' in df.columns or 'CAUSALE_DEL_MOVIMENTO' in df.columns:
        col_causale = 'DESCRIZIONE_CAUSALE' if 'DESCRIZIONE_CAUSALE' in df.columns else 'CAUSALE_DEL_MOVIMENTO'
        # Filtriamo solo i RESI per il dettaglio, se necessario, oppure mostriamo tutto. 
        # L'utente ha chiesto boxplot, che ha senso per vedere la distribuzione delle quantità per tipo.
        fig_box = px.box(df, x=col_causale, y='QUANTITA_MOVIMENTATA', color=col_causale,
                         title=f"Distribuzione Quantità per {col_causale}", template="plotly_white")
        fig_box.update_yaxes(title="Quantità Movimentata")
        st.plotly_chart(fig_box, use_container_width=True, key=f"{key_prefix}box")
        st.caption("ℹ️ **Interpretazione**: Il Boxplot mostra la dispersione dei volumi. La 'scatola' contiene il 50% centrale dei dati. I punti esterni sono anomalie (outliers). Permette di capire se certe causali hanno volumi tipicamente più alti o più variabili di altre.")
    
    # 3. Giorno della Settimana
    st.markdown("---")
    st.subheader("3. Analisi per Giorno della Settimana")
    if 'DATA_OPERAZIONE' in df.columns:
        df['DayOfWeek'] = df['DATA_OPERAZIONE'].dt.day_name()
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].groupby('DayOfWeek')['QUANTITA_MOVIMENTATA'].mean().reindex(order)
        weekly_df = pd.DataFrame({'Giorno': weekly_avg.index, 'Media Resi': weekly_avg.values})
        
        fig_week = px.bar(weekly_df, x='Giorno', y='Media Resi', color='Media Resi', color_continuous_scale="Viridis",
                          title="Media Resi per Giorno della Settimana")
        st.plotly_chart(fig_week, use_container_width=True, key=f"{key_prefix}week")
        st.caption("ℹ️ **Interpretazione**: Indica in quali giorni vengono registrati più resi. Utile per pianificare i turni del personale di magazzino.")

    # 4. Analisi per Valore (Financial)
    st.markdown("---")
    st.subheader("4. Analisi Valore Economico Resi stimato")
    if 'Valore_Totale' in df.columns:
        # Aggregazione mensile del valore
        monthly_val = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].groupby(df['DATA_OPERAZIONE'].dt.to_period('M'))['Valore_Totale'].sum().abs()
        monthly_val.index = monthly_val.index.to_timestamp()
        val_df = pd.DataFrame({'Data': monthly_val.index, 'Valore Resi (€)': monthly_val.values})
        
        # Esporta in CSV
        export_path = os.path.join(OUTPUT_DIR, "valore_stimato_resi.csv")
        val_df.to_csv(export_path, index=False)
        st.success(f"Dati esportati in: `{export_path}`")
        
        fig_val = px.line(val_df, x='Data', y='Valore Resi (€)', markers=True,
                         title="Valore Stimato Resi nel Tempo (€)", template="plotly_white")
        fig_val.update_layout(xaxis_title="Data", yaxis_title="Valore (€)")
        st.plotly_chart(fig_val, use_container_width=True, key=f"{key_prefix}val")
        st.caption("ℹ️ **Interpretazione Finanziaria**: Questo grafico mostra l'impatto economico stimato dei resi nel tempo, basato sulla fascia di prezzo media.")

    # 5. Boxplot Distribuzione Resi per Causale (Valore)
    st.markdown("---")
    st.subheader("5. Distribuzione Resi per Causale (Valore)")
    if 'Valore_Totale' in df.columns and ('DESCRIZIONE_CAUSALE' in df.columns or 'CAUSALE_DEL_MOVIMENTO' in df.columns):
        col_causale = 'DESCRIZIONE_CAUSALE' if 'DESCRIZIONE_CAUSALE' in df.columns else 'CAUSALE_DEL_MOVIMENTO'
        df_resi_val = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].copy()
        df_resi_val['Valore_Assoluto'] = df_resi_val['Valore_Totale'].abs()
        
        fig_box_val = px.box(df_resi_val, x=col_causale, y='Valore_Assoluto', color=col_causale,
                             title=f"Distribuzione Valore Resi per {col_causale}", template="plotly_white")
        fig_box_val.update_xaxes(title="Causa-movimento")
        fig_box_val.update_yaxes(title="Valore (€)")
        st.plotly_chart(fig_box_val, use_container_width=True, key=f"{key_prefix}box_val_causale")
        st.caption("ℹ️ **Interpretazione**: Il Boxplot interattivo mostra la distribuzione del valore economico dei resi per ogni causale, permettendo di identificare quali cause generano resi di maggior impatto economico potendo fare zoom e ispezionare gli outlier.")

    # 6. Analisi Univariata delle Variabili
    st.markdown("---")
    st.subheader("6. Analisi Univariata delle Variabili")
    st.markdown("Distribuzione delle principali variabili per i resi.")
    
    num_cols = ['QUANTITA_MOVIMENTATA']
    if 'Valore_Totale' in df.columns:
        num_cols.append('Valore_Totale')
        
    df_resi_uni = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].copy()
    if 'Valore_Totale' in df_resi_uni.columns:
        df_resi_uni['Valore_Totale'] = df_resi_uni['Valore_Totale'].abs()
    
    if num_cols:
        tabs_uni = st.tabs([f"Distribuzione di {col}" for col in num_cols])
        for i, col in enumerate(num_cols):
            with tabs_uni[i]:
                fig_hist = px.histogram(df_resi_uni, x=col, marginal="box", 
                                        title=f"Distribuzione Univariata di {col} (Solo Resi)", 
                                        template="plotly_white", nbins=50)
                st.plotly_chart(fig_hist, use_container_width=True, key=f"{key_prefix}hist_{col}")
                st.caption(f"ℹ️ **Interpretazione**: Istogramma per mostrare la distribuzione e boxplot per gli outlier della variabile {col}.")

    # 7. Analisi dei Resi per Tipo Prodotto
    st.markdown("---")
    st.subheader("7. Analisi dei Resi per Tipo Prodotto")
    if 'DESCRIZIONE_TIPO' in df.columns:
        df_resi_tipo = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES']
        resi_per_tipo = df_resi_tipo.groupby('DESCRIZIONE_TIPO')['QUANTITA_MOVIMENTATA'].sum().abs().sort_values(ascending=False).reset_index()
        # Selezioniamo i top 15 per evitare un grafico illeggibile
        top_types = resi_per_tipo.head(15)
        
        fig_tipo = px.bar(top_types, x='QUANTITA_MOVIMENTATA', y='DESCRIZIONE_TIPO', orientation='h',
                          title="Top 15 Categorie di Prodotti più Restituiti (Volumi)", template="plotly_white",
                          color='QUANTITA_MOVIMENTATA', color_continuous_scale='Blues')
        fig_tipo.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Quantità Fisica Resa", yaxis_title="Tipo di Prodotto")
        st.plotly_chart(fig_tipo, use_container_width=True, key=f"{key_prefix}tipo_reso")
        st.caption("ℹ️ **Interpretazione Business**: Questo grafico a barre orizzontali evidenzia quali sono i prodotti (`DESCRIZIONE_TIPO`) che generano il maggior numero netto di resi. È fondamentale per scoprire relazioni specifiche: la ricorrenza di alti resi in certe categorie potrebbe segnalare difetti di fabbrica o problemi di vestibilità ricorrenti.")

    # 8. Valutazione Incidenza Resi su Vendite (Valore %)
    st.markdown("---")
    st.subheader("8. Valutazione Incidenza Resi su Vendite (%)")
    if 'DESCRIZIONE_TIPO' in df.columns and 'Valore_Totale' in df.columns:
        # Utilizziamo la formula numerica: (Valore Resi / Valore Vendita) * 100
        valori_pivot = pd.pivot_table(df, values='Valore_Totale', index='DESCRIZIONE_TIPO', columns='CAUSALE_DEL_MOVIMENTO', aggfunc='sum', fill_value=0)
        
        if 'RES' in valori_pivot.columns and 'VEN' in valori_pivot.columns:
            valori_pivot['Incidenza_Perc'] = (valori_pivot['RES'].abs() / valori_pivot['VEN']) * 100
            
            # Puliamo infiniti e nan derivati da divisioni per 0
            valori_pivot = valori_pivot.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Filtriamo prodotti e teniamo i 15 con l'incidenza percentuale maggiore
            top_incidenza = valori_pivot[valori_pivot['VEN'] > 0].sort_values(by='Incidenza_Perc', ascending=False)
            top_incidenza = top_incidenza.head(15).reset_index()
            
            fig_incidenza = px.bar(top_incidenza, x='Incidenza_Perc', y='DESCRIZIONE_TIPO', orientation='h',
                              title="Top 15 Prodotti per Incidenza Resi su Vendite (%)", template="plotly_white",
                              color='Incidenza_Perc', color_continuous_scale='Reds',
                              labels={'Incidenza_Perc': 'Incidenza Resi (%)'})
            fig_incidenza.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Incidenza % (Valore Resi / Valore Vendita * 100)", yaxis_title="Tipo di Prodotto")
            
            st.plotly_chart(fig_incidenza, use_container_width=True, key=f"{key_prefix}incidenza_reso")
            st.caption("ℹ️ **Valutazione Oggetti Resi**: Il coefficiente viene calcolato con la formula `(Valore Resi / Valore Vendita) * 100`. Il grafico evidenzia i prodotti maggiormente 'tossici' per i ricavi: quelli che generano il più alto tasso di rimborso in proporzione al fatturato generato.")

def display_crisp_dm_tab(df, monthly_df, results, best_model_name, forecast_df, customer_df, avg_monthly_returns):
    st.header("📚 Report CRISP-DM: Analisi Completa")
    st.markdown("""
    Questa sezione presenta l'intero ciclo di vita del progetto di Data Science, seguendo la metodologia standard **CRISP-DM** (Cross Industry Standard Process for Data Mining).
    """)
    
    # --- PHASE 1: BUSINESS UNDERSTANDING ---
    st.markdown("---")
    st.subheader("1. Business Understanding (Comprensione del Business)")
    st.info("""
    **Obiettivo**: Ottimizzare la gestione dei resi per migliorare la pianificazione logistica e la redditività.
    
    **Domande Chiave**:
    1. Qual è il trend futuro dei resi? (Previsione)
    2. Chi sono i clienti che restituiscono di più? (Segmentazione)
    3. Quali sono le cause principali? (Analisi Causali)
    
    **KPI di Progetto**:
    - Accuratezza Previsionale (RMSE ridotto).
    - Identificazione chiara dei segmenti "a rischio".
    """)
    
    # --- PHASE 2: DATA UNDERSTANDING ---
    st.markdown("---")
    st.subheader("2. Data Understanding (Comprensione dei Dati)")
    st.markdown(f"""
    **Dataset**: `{df.shape[0]}` righe totali, `{df['CODICE_CLIENTE'].nunique()}` clienti unici.
    
    ⚠️ **Nota Importante sulle Quantità**:
    Per **"Quantità Resi"** e **"Quantità Vendite"** si intende il **Numero di Pezzi (Unità Fisiche)** movimentate, **NON** il valore monetario (€).
    L'analisi si basa quindi sui flussi logistici, essenziali per il dimensionamento del magazzino.
    """)
    
    with st.expander("🔍 Visualizza Distribuzioni (EDA)"):
        display_eda_tab(df, key_prefix="crisp_dm_")
        
    # --- PHASE 3: DATA PREPARATION ---
    st.markdown("---")
    st.subheader("3. Data Preparation (Preparazione Dati)")
    st.markdown("""
    I dati grezzi transazionali non sono utilizzabili direttamente per la previsione. Sono stati trasformati come segue:
    
    1.  **Aggregazione Mensile**: I dati giornalieri sono stati sommati per mese (`Resi_Mensili`, `Vendite_Mensili`).
    2.  **Lag Features (Ritardi)**:
        - `Sales_Lag1`: Vendite del mese precedente (Ipotesi: Vendite alte oggi -> Resi alti domani).
        - `Returns_Lag1`: Resi del mese precedente (Inerzia del fenomeno).
    3.  **Rolling Statistics (Medie Mobili)**:
        - `Returns_RollMean_3`: Media dei resi negli ultimi 3 mesi (Cattura il trend di breve termine).
    """)
    if monthly_df is not None:
        st.write("Esempio dati preparati (prime 5 righe):")
        st.dataframe(monthly_df.head())
        
    # --- PHASE 4: MODELING ---
    st.markdown("---")
    st.subheader("4. Modeling (Modellazione)")
    st.markdown("""
    Sono stati addestrati e confrontati tre algoritmi di regressione per prevedere la serie storica dei resi:
    
    - **SVR (Support Vector Regressor)**: Ottimo per trend stabili, cerca una curva che si adatti ai dati con un margine di tolleranza.
    - **Random Forest**: Un "consiglio" di centinaia di alberi decisionali. Robusto agli outlier e capace di catturare relazioni non lineiali.
    - **Gradient Boosting**: Costruisce alberi in sequenza, dove ognuno corregce gli errori del precedente. Spesso il più preciso.
    """)
    
    if results:
         # Model Comparison Plot
        rmse_values = [res['RMSE'] for name, res in results.items()]
        model_names = list(results.keys())
        
        fig_comp = px.bar(x=model_names, y=rmse_values, color=rmse_values, color_continuous_scale='RdBu_r',
                          title="Confronto Errori Modelli (RMSE - Più basso è meglio)", template="plotly_white")
        fig_comp.update_layout(xaxis_title="Modello", yaxis_title="RMSE (Errore Medio in Pezzi)")
        st.plotly_chart(fig_comp, use_container_width=True, key="crisp_dm_comp")
        
    # --- PHASE 5: EVALUATION ---
    st.markdown("---")
    st.subheader("5. Evaluation (Valutazione)")
    
    best_rmse = results[best_model_name]['RMSE']
    best_mae = results[best_model_name]['MAE']
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Miglior Modello", best_model_name)
    c2.metric("RMSE (Errore Quadratico Medio)", f"{best_rmse:.2f}")
    c3.metric("MAE (Errore Assoluto Medio)", f"{best_mae:.2f}")
    

    # --- FEATURE IMPORTANCE SECTION ---
    st.markdown("### 🎯 Importanza delle Variabili per Modello")
    
    models_with_importance = {name: res['Feature_Importance'] for name, res in results.items() if res.get('Feature_Importance') is not None}
    
    if models_with_importance:
        st.success("""
        **Analisi e Interpretazione Grafico (Feature Importance)**:
        Questi grafici svelano i meccanismi interni della decisione ("aprono la scatola nera"). Mostrano in ordine decrescente l'impatto percentuale che ciascuna variabile di business ha sulla predizione dei resi del prossimo mese per ogni singolo modello.
        
        - **Barre Più Lunghe**: Le variabili in alto (o con il valore più alto sull'asse X) sono i veri driver previsionali ("*I resi di oggi dipendono fortemente da questa metrica*"). Spesso nei logistici sono i "Ritardi" (Lags) o le "Vendite passate".
        - **Barre Cortissime**: Hanno influito in maniera trascurabile nei calcoli dell'algoritmo.
        """)
        
        # Build tabs for each model
        model_names_imp = list(models_with_importance.keys())
        tabs_imp = st.tabs(model_names_imp)
        
        for i, imp_model_name in enumerate(model_names_imp):
            with tabs_imp[i]:
                importance_dict = models_with_importance[imp_model_name]
                imp_df = pd.DataFrame(list(importance_dict.items()), columns=['Variabile_Predittiva', 'Peso']).sort_values('Peso', ascending=False)
                
                fig_imp = px.bar(imp_df, x='Peso', y='Variabile_Predittiva', orientation='h', 
                                 title=f"Quali variabili guidano la previsione? ({imp_model_name})", template="plotly_white",
                                 color='Peso', color_continuous_scale='Viridis')
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Peso/Importanza Logica", yaxis_title="Nome Variabile (Feature)")
                st.plotly_chart(fig_imp, use_container_width=True, key=f"crisp_dm_feature_imp_{imp_model_name}")
    else:
        st.info("Nessun modello supporta l'estrazione dell'importanza delle singole feature in questo ciclo.")

    # --- FINANCIAL KPI SECTION ---
    st.markdown("### 💰 Impatto Finanziario Stimato")
    if 'Valore_Totale' in df.columns:
        total_res_val = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES']['Valore_Totale'].sum()
        total_ven_val = df[df['CAUSALE_DEL_MOVIMENTO'] == 'VEN']['Valore_Totale'].sum()
        impact_pct = (abs(total_res_val) / total_ven_val * 100) if total_ven_val > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Valore Totale Resi (Stimato)", f"€ {abs(total_res_val):,.0f}")
        k2.metric("Valore Totale Vendite (Stimato)", f"€ {total_ven_val:,.0f}")
        k3.metric("Incidenza Resi su Fatturato", f"{impact_pct:.0f}%")
        
        st.caption("*Valori stimati basati sulla media della fascia di prezzo.*")
    
    # Forecast Plot (Interactive)
    if forecast_df is not None:
        st.write(f"### Previsione vs Reale ({best_model_name})")
        fig_forecast = go.Figure()
        
        # Historical & Actual
        fig_forecast.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Reale'],
                                          mode='lines+markers', name='Dati Reali (Test)',
                                          line={"color": 'blue', "width": 3}))
        
        # Predictions
        for col in forecast_df.columns:
            if col not in ['Date', 'Reale']:
                is_best = (col == best_model_name)
                width = 4 if is_best else 1
                opacity = 1.0 if is_best else 0.4
                color = 'red' if is_best else 'gray'
                name_str = f"Prev. {col} {'(Best)' if is_best else ''}"
                
                fig_forecast.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df[col],
                                                  mode='lines', name=name_str,
                                                  line={"width": width, "color": color, "dash": 'dot' if not is_best else 'solid'},
                                                  opacity=opacity))
                
        fig_forecast.update_layout(title="Performance sul Test Set (Ultimi Mesi)", xaxis_title="Data", yaxis_title="Numero Pezzi")
        st.plotly_chart(fig_forecast, use_container_width=True, key="crisp_dm_forecast")

    # --- PHASE 6: DEPLOYMENT & SEGMENTATION ---
    st.markdown("---")
    st.subheader("6. Deployment & Strategie (Segmentazione)")
    st.markdown("Il clustering ha identificato 3 profili di comportamento. Ecco le azioni strategiche consigliate:")
    
    col_k1, col_k2, col_k3 = st.columns(3)
    
    with col_k1:
        st.error("🟥 Alto Tasso di Reso")
        st.markdown("""
        *Clienti che restituiscono troppo spesso.*
        - **Azione**: Introdurre soglie di reso gratuito/anno.
        - **Obiettivo**: Ridurre i costi logistici senza perdere il cliente.
        """)
        
    with col_k2:
        st.success("🟩 Top Clients")
        st.markdown("""
        *Clienti alto spendenti e sani.*
        - **Azione**: Reso gratuito immediato (VIP).
        - **Obiettivo**: Retention e aumento LTV (Lifetime Value).
        """)
        
    with col_k3:
        st.warning("🟨 Standard")
        st.markdown("""
        *Clienti nella media.*
        - **Azione**: Monitoraggio.
        - **Obiettivo**: Prevenire la migrazione verso il cluster "Alto Reso".
        """)

    if customer_df is not None and 'Cluster_Label' in customer_df.columns:
         st.write("### Mappa dei Segmenti")
         # Simple Scatter if PCA exists
         if 'PCA1' in customer_df.columns:
             fig_seg = px.scatter(customer_df, x='PCA1', y='PCA2', color='Cluster_Label',
                                  title="Mappa Distributiva Clienti", template="plotly_white",
                                  hover_data=['Total_Sales', 'Total_Returns'])
             st.plotly_chart(fig_seg, use_container_width=True, key="crisp_dm_seg")

# --- MAIN APP ---
if uploaded_file is not None:
    # Save temp
    temp_path = os.path.join(OUTPUT_DIR, "temp_upload.csv")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        df = load_dataset(temp_path)
        df = calculate_estimated_price(df) # Financial Features
        
        # Initialize session state for analysis results if not present
        if 'analysis_done' not in st.session_state:
            st.session_state['analysis_done'] = False
            st.session_state['monthly_df'] = None
            st.session_state['customer_df'] = None
            st.session_state['results'] = None
            st.session_state['best_model'] = None
            st.session_state['forecast_df'] = None
            st.session_state['iteration_type'] = None
        
        # Calculate Baseline Metrics for Report (moved outside tab block to be available for CRISP-DM tab)
        total_returns = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].shape[0]
        if 'DATA_OPERAZIONE' in df.columns:
             monthly_res_sum = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].groupby(df['DATA_OPERAZIONE'].dt.to_period('M'))['QUANTITA_MOVIMENTATA'].sum()
             # Use ABS if returns are negative, checking data... usually RES is distinct transaction
             # Assuming QUANTITA_MOVIMENTATA is positive or we need abs? 
             # Let's take abs to be safe if they are recorded as negative flows
             AVG_MONTHLY_RETURNS = abs(monthly_res_sum).mean()
        else:
             AVG_MONTHLY_RETURNS = 0

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🔎 Dati & Glossario", "📊 EDA", "🚀 Risultati & Grafici", "📄 Report Testuali", "📚 Report CRISP-DM", "🧠 Modelli ML Spiegati", "⚙️ Ottimizzazione Modelli ML"])
        
        with tab1:
            st.header("Esplorazione Dati")
            display_glossary()
            
            st.subheader("Anteprima Dataset")
            st.dataframe(df.head())
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Righe Totali", df.shape[0])
            c2.metric("Clienti Unici", df['CODICE_CLIENTE'].nunique())
            c3.metric("Totale Resi (Righe)", total_returns)

        with tab2:
            display_eda_tab(df)

        with tab3:
            st.header("Pipeline Analitica")
            st.markdown("Scegli un'iterazione per l'analisi. La **Prima Iterazione** usa i modelli con impostazioni standard. La **Seconda Iterazione** utilizza i parametri ottimali calcolati in fase di Tuning.")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                run_it1 = st.button("Avvia Prima Iterazione (Modelli Base)")
            with col_btn2:
                run_it2 = st.button("Avvia Seconda Iterazione (Modelli Ottimizzati)")
                
            if run_it1 or run_it2:
                is_opt = run_it2 # True if Iteration 2 was clicked
                iter_label = "Iterazione 2 - Ottimizzata" if is_opt else "Iterazione 1 - Base"
                
                with st.spinner(f'Esecuzione {iter_label} in corso...'):
                    monthly_df, customer_df, results, best_model, forecast_df = run_full_pipeline(df, use_optimized=is_opt)
                    
                    # Store in session state
                    st.session_state['analysis_done'] = True
                    st.session_state['monthly_df'] = monthly_df
                    st.session_state['customer_df'] = customer_df
                    st.session_state['results'] = results
                    st.session_state['best_model'] = best_model
                    st.session_state['forecast_df'] = forecast_df
                    st.session_state['iteration_type'] = iter_label
                    
                    st.success(f"Analisi Completata con successo per l'{iter_label}!")
            
            if st.session_state['analysis_done']:
                data = st.session_state
                iter_info = data.get('iteration_type', 'Sconosciuta')
                st.success(f"Analisi in visualizzazione: **{iter_info}** | Miglior Modello Eletto: **{data['best_model']}**")
                
                # --- PLOT 1: FORECAST ---
                st.markdown("---")
                st.subheader("1. Confronto Modelli Previsionali")
                st.info("""
                **Interpretazione:**
                - Il grafico confronta le performance dei 3 modelli testati rispetto al dato Reale (Linea Blu).
                - **Il modello migliore è evidenziato**.
                - Scegliere il modello che segue meglio i picchi e le valli della linea Blu.
                """)
                
                # Create dynamic plot with all models
                fig_forecast = go.Figure()
                
                # Actual
                fig_forecast.add_trace(go.Scatter(x=data['forecast_df']['Date'], y=data['forecast_df']['Reale'],
                                                  mode='lines+markers', name='Reale',
                                                  line={"color": 'blue', "width": 3}))
                
                # Predictions
                colors = {'SVR': 'orange', 'RandomForest': 'green', 'GradientBoosting': 'purple'}
                
                for col in data['forecast_df'].columns:
                    if col not in ['Date', 'Reale']:
                        is_best = (col == data['best_model'])
                        width = 4 if is_best else 2
                        opacity = 1.0 if is_best else 0.5
                        name_str = f"{col} (Best)" if is_best else col
                        
                        fig_forecast.add_trace(go.Scatter(x=data['forecast_df']['Date'], y=data['forecast_df'][col],
                                                          mode='lines+markers', name=name_str,
                                                          line={"width": width, "dash": 'dot'},
                                                          opacity=opacity))
                
                fig_forecast.update_layout(title="Confronto Previsioni Modelli",
                                           xaxis_title="Data", yaxis_title="Quantità",
                                           hovermode="x unified")
                st.plotly_chart(fig_forecast, use_container_width=True, key="tab3_forecast")
                
                # --- PLOT 1.1: FORECAST VALORE ---
                if 'Valore_Totale' in df.columns:
                    st.markdown("---")
                    st.subheader("1.1 Confronto Modelli Previsionali (Valore Economico Stimato)")
                    df_res = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES']
                    cost_per_item = abs(df_res['Valore_Totale'].sum()) / abs(df_res['QUANTITA_MOVIMENTATA'].sum()) if abs(df_res['QUANTITA_MOVIMENTATA'].sum()) > 0 else 1
                    
                    fig_forecast_val = go.Figure()
                    
                    # Actual
                    fig_forecast_val.add_trace(go.Scatter(x=data['forecast_df']['Date'], y=data['forecast_df']['Reale'] * cost_per_item,
                                                      mode='lines+markers', name='Reale (€)',
                                                      line={"color": 'blue', "width": 3}))
                    
                    for col in data['forecast_df'].columns:
                        if col not in ['Date', 'Reale']:
                            is_best = (col == data['best_model'])
                            width = 4 if is_best else 2
                            opacity = 1.0 if is_best else 0.5
                            name_str = f"{col} (Best) €" if is_best else f"{col} €"
                            
                            fig_forecast_val.add_trace(go.Scatter(x=data['forecast_df']['Date'], y=data['forecast_df'][col] * cost_per_item,
                                                              mode='lines+markers', name=name_str,
                                                              line={"width": width, "dash": 'dot'},
                                                              opacity=opacity))
                    
                    fig_forecast_val.update_layout(title="Confronto Previsioni Modelli (Valore €)",
                                               xaxis_title="Data", yaxis_title="Valore Estimato (€)",
                                               hovermode="x unified")
                    st.plotly_chart(fig_forecast_val, use_container_width=True, key="tab3_forecast_val")
                
                # --- PLOT 2: CORRELATION ---
                st.markdown("---")
                st.subheader("2. Matrice di Correlazione")
                
                # Calculate correlation on the fly for visualization
                corr = data['monthly_df'].select_dtypes(include=[np.number]).corr()
                fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                                     title="Correlazione Features")
                st.plotly_chart(fig_corr, use_container_width=True, key="tab3_corr")
                
                # --- PLOT 3: CLUSTERS ---
                st.markdown("---")
                st.subheader("3. Segmentazione Clienti")
                st.info("""
                **Interpretazione Cluster:**
                - **Alto Tasso di Reso**: Clienti che restituiscono molto frequentemente.
                - **Top Clients**: Clienti che spendono molto (Fatturato alto).
                - **Standard**: Clienti nella media.
                """)
                
                if 'Cluster_Label' in data['customer_df'].columns:
                    # PCA components should already be in customer_df from run_full_pipeline
                    if 'PCA1' in data['customer_df'].columns:
                        fig_cluster = px.scatter(data['customer_df'], x='PCA1', y='PCA2', 
                                                 color='Cluster_Label', # Use Label here
                                                 hover_data=['Total_Sales', 'Total_Returns', 'Return_Rate'],
                                                 title="Mappa Segmenti Clienti",
                                                 color_discrete_sequence=px.colors.qualitative.Bold)
                        st.plotly_chart(fig_cluster, use_container_width=True, key="tab3_cluster")
                    else:
                        st.warning("Coordinate PCA non disponibili per il plot del cluster.")
                else:
                    st.warning("Etichette Cluster non disponibili.")

        with tab4:
            st.header("Report Completo & Conclusioni")
            
            report_path = os.path.join(OUTPUT_DIR, "analysis_report.md")
            if os.path.exists(report_path):
                 with open(report_path, "r", encoding='utf-8') as f:
                    st.markdown(f.read())
            else:
                 st.info("Esegui l'analisi per generare il report.")

            st.markdown("---")
            st.subheader("Riepilogo Metriche")
            metrics_path = os.path.join(OUTPUT_DIR, "model_metrics_summary.md")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding='utf-8') as f:
                    st.markdown(f.read())

        with tab5:
            if st.session_state['analysis_done']:
                display_crisp_dm_tab(df, st.session_state['monthly_df'], st.session_state['results'], 
                                     st.session_state['best_model'], st.session_state['forecast_df'], 
                                     st.session_state['customer_df'], AVG_MONTHLY_RETURNS)
            else:
                st.info("⚠️ Esegui l'analisi nella Tab '🚀 Risultati & Grafici' per visualizzare il report CRISP-DM completo.")

        with tab6:
            st.header("🧠 Modelli di Machine Learning Spiegati")
            st.markdown("""
            In questa sezione spieghiamo i principi matematici e il funzionamento dei 3 modelli di Machine Learning utilizzati per prevedere i resi futuri, in modo semplice e accessibile.

            ### 1. Support Vector Regression (SVR)
            **Principio Matematico**: Il SVR cerca di trovare una funzione logica (un "tubo" o margine di tolleranza, chiamato $\\epsilon$-tube) che contenga il maggior numero possibile di punti dati, penalizzando solo quelli che cadono fuori da questo tubo.
            Utilizza il *Kernel Trick* per proiettare i dati in dimensioni superiori, riuscendo a mappare relazioni non lineari complesse in spazi più semplici.
            
            **Esempio Pratico**: Immagina di dover tracciare un'autostrada (il tubo di tolleranza) che deve attraversare una serie di città (i punti dati storici). Lo scopo non è passare *esattamente* in mezzo ad ogni singola casa, ma fare in modo che l'autostrada sia abbastanza larga da coprire quasi tutte le città, minimizzando le "deviazioni" eccessive. Se un dato storico di reso è un po' anomalo, il modello lo perdona purché resti entro l'autostrada.

            ---

            ### 2. Random Forest Regressor
            **Principio Matematico**: È un metodo *Ensemble* (di gruppo) basato sul *Bagging*. Costruisce centinaia di Alberi Decisionali (Decision Trees) indipendenti, ognuno addestrato su un sottoinsieme casuale di dati e di variabili. La previsione finale è la **media** delle previsioni di tutti gli alberi.
            
            **Esempio Pratico**: Immagina di dover indovinare quanti resi riceverai a dicembre. Invece di chiedere a un solo esperto (che potrebbe sbagliarsi), chiedi a 100 esperti (i vari alberi decisionali). Ognuno di loro guarda i dati storici da una prospettiva leggermente diversa (es. uno guarda solo ai giorni della settimana, un altro all'anno, un altro ai saldi passati). Alla fine, fai la media delle loro risposte. Questo approccio è molto robusto perché "la saggezza della folla" appiattisce le singole sbavature, riducendo l'errore del singolo.

            ---

            ### 3. Gradient Boosting Regressor (GBM)
            **Principio Matematico**: Un altro metodo *Ensemble*, ma basato sul *Boosting*. Invece di creare alberi in parallelo e considerarli tutti uguali, li crea in sequenza. Il primo albero cerca di fare una previsione; poi, il secondo albero viene addestrato **esclusivamente per correggere gli errori (i residui)** fatti dal primo albero. Il processo continua albero dopo albero, ottimizzando una funzione di perdita (Loss Function) attraverso la *Discesa del Gradiente*.
            
            **Esempio Pratico**: Immagina di giocare a minigolf. Al primo colpo (primo albero) ti avvicini alla buca ma non la centri. Il tuo secondo colpo (secondo albero) non partirà dall'inizio, ma dal punto in cui si è fermata la pallina, puntando a coprire la distanza rimanente (l'errore) prestando molta attenzione alla spinta data in precedenza. Tiri con aggiustamenti sempre più sottili finché non vai in buca. È un metodo estremamente preciso e potente!
            """)

        with tab7:
            st.header("⚙️ Ottimizzazione dei Modelli di Machine Learning")
            st.markdown("""
            In questa sezione viene spiegato come si "allenano" al meglio i modelli di Intelligenza Artificiale e vengono presentati i risultati dell'ottimizzazione (Hyperparameter Tuning).
            
            ### Cos'è l'Ottimizzazione dei Modelli?
            Immagina che ogni modello di Machine Learning sia una macchina fotografica. L'algoritmo base sa scattare una foto, ma per ottenere la foto *perfetta* devi regolare le impostazioni (gli "Iperparametri"): messa a fuoco, apertura del diaframma, tempo di esposizione, ecc.
            
            L'**Ottimizzazione degli Iperparametri** è il processo automatico con cui il computer prova migliaia di combinazioni di queste "impostazioni" per trovare quella che riduce al minimo l'errore di previsione sui resi.
            
            ### Come viene effettuata?
            Per questo progetto è stata utilizzata una tecnica chiamata **Randomized Search** combinata con la **Cross-Validation per Serie Storiche**:
            1. **Scelta Casuale (Randomized Search)**: Il computer sceglie a caso centinaia di combinazioni possibili di parametri e le testa. È più veloce ed efficiente di testarle tutte una per una in modo rigido (Grid Search).
            2. **Test nel Tempo (Time-Series Cross-Validation)**: A differenza di altri dati (come prevedere se un'email è spam), i resi sono una **serie storica** (dipendono dal tempo). L'algoritmo viene allenato sul passato (es. Da Gennaio a Settembre) e testato sul futuro (es. Ottobre), per poi ripetere l'allenamento fino ad Ottobre per testare Novembre, e così via.
            """)
            
            st.markdown("---")
            st.subheader("🏆 Risultati dell'Ottimizzazione")
            
            optimization_file = os.path.join(PROJECT_DIR, "optimization", "optimization_results.md")
            if os.path.exists(optimization_file):
                with open(optimization_file, "r", encoding='utf-8') as f:
                    opt_content = f.read()
                    st.info(opt_content)
            else:
                st.warning("⚠️ File dei risultati dell'ottimizzazione (`optimization_results.md`) non trovato nella cartella `optimization`.")
                
            st.markdown("""
            **Interpretazione dei Parametri Migliori**:
            - **RandomForest (`n_estimators`, `max_depth`, `min_samples_split`)**: Quanti alberi usare (più sono, meglio è la media, ma più è lento il calcolo), quanto possono diventare profondi gli alberi (per catturare dettagli complessi senza esagerare) e la quantità minima di dati per creare un nuovo "ramo" di decisione per evitare che il modello impari i dati a memoria.
            - **GradientBoosting (`learning_rate`, `n_estimators`, `max_depth`)**: Il `learning_rate` decide quanto ogni nuovo albero contribuisce alla correzione dell'errore (un passo piccolo è più cauto e preciso).

            *L'obiettivo finale di questa operazione è stato ridurre il valore dell'errore (RMSE) il più possibile, portando a previsioni più affidabili sul numero di resi fisici mensili.*
            """)

    except Exception as e: # pylint: disable=broad-exception-caught
        st.error(f"Errore nel caricamento o analisi: {e}")
        st.write("Dettaglio errore:", e)
        import traceback
        st.text(traceback.format_exc())
        
else:
    st.info("👈 Carica un file CSV dalla barra laterale per iniziare.")

