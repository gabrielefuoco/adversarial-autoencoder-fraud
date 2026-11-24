# Adversarial Autoencoder per il rilevamento di frodi finanziarie

![Spazio latente dell'autoencoder standard](Relazione%20Finale/assets/tsne_standard_ae.png)

Riproduzione ed estensione di un approccio con autoencoder avversario all'anomaly detection finanziaria: lo spazio latente appreso da un autoencoder viene usato come spazio di feature per classificatori supervisionati, e ogni affermazione è misurata contro una baseline. Il risultato principale è che, su questo dataset, la regolarizzazione avversaria non ripaga la sua complessità.

## Indice

- [Risultati in breve](#risultati-in-breve)
- [Come funziona](#come-funziona)
- [Decisioni](#decisioni)
- [Mappa del repository](#mappa-del-repository)
- [Eseguire gli esperimenti](#eseguire-gli-esperimenti)
- [Limiti](#limiti)
- [Documentazione](#documentazione)

## Risultati in breve

AUC multi-classe a parità di dimensionalità latente ($d=10$), classificatori addestrati sui vettori latenti dell'autoencoder:

| Classificatore | AE avversario ($z_{AAE}$) | AE standard ($z_{AE}$) | Delta |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 0.9687 | **0.9841** | +1.54% per l'AE standard |
| **SVM (kernel RBF)** | 0.9950 | **0.9962** | +0.12% per l'AE standard |

L'ablation mantiene encoder, decoder, layer, neuroni e attivazioni identici e rimuove solo il discriminatore e la fase avversaria: la differenza è quindi attribuibile alla sola regolarizzazione dello spazio latente. Forzare lo spazio latente verso una mistura di Gaussiane ha reso la classificazione leggermente peggiore, non migliore: su questo dataset la topologia naturale della rete separa le frodi meglio di una imposta.

## Come funziona

### Il dataset

Una derivazione di **PaySim**, simulatore di transazioni di mobile money: **532.909 transazioni** con un tasso di frode dell'**1-2%** e **618 feature**. La difficoltà sta nella mescolanza: codici categorici ad alta cardinalità (conto di contabilità generale `HKONT`, posting key `BSCHL`, centro di profitto `PRCTR`) accanto a importi continui (`DMBTR`, `WRBTR`). Codificare quelle colonne in modo ingenuo produce vettori sparsi ingestibili: il lavoro vero è nella pre-elaborazione.

### La rappresentazione latente

Tre reti in PyTorch — encoder, decoder, discriminatore. L'encoder mappa una transazione in un **vettore latente a 10 dimensioni**; il decoder ricostruisce l'ingresso; il discriminatore spinge la distribuzione latente aggregata verso una mistura di Gaussiane, così il codice latente è insieme ricostruttivo e regolarizzato. I vettori latenti diventano poi le **feature** per SVM e Random Forest: è l'integrazione proposta dal paper originale.

### La metrica

**MAUC**, l'estensione multi-classe dell'AUC di Hand & Till (2001), media pairwise dell'approssimazione probabilistica dell'AUC, implementata in `MAUC.py`. La scelta conta: su un dataset licito al 98-99% l'accuracy non significa niente (un classificatore che risponde sempre «regolare» supera 0,99). L'AUC misura se le frodi sono *ordinate* sopra le transazioni lecite, che è quello che serve a una coda di revisione.

### Robustezza allo sbilanciamento

Uno stress test fa variare il tasso di frode nel training set:

- **sotto l'1%** il modello tiene, con varianza leggermente più alta;
- **intorno al 2%** le prestazioni hanno un picco: una piccola contaminazione sembra aiutare il discriminatore a definire i confini della distribuzione prior, funzionando da regolarizzatore implicito;
- **oltre il 5%** crolla, e al **10%** il MAUC dei classificatori lineari scende sotto **0,85**.

La causa è la **pollution**: quando le anomalie diventano abbastanza frequenti, l'autoencoder le apprende come normali e le ricostruisce con errore basso, così il punteggio di ricostruzione perde potere discriminante. Operativamente il metodo è valido solo finché le frodi restano rare: un picco improvviso di frodi può rendere il sistema cieco, ed è il motivo per cui un deploy in produzione richiede un monitoraggio del drift di distribuzione accanto al modello.

![Effetto pollution: MAUC contro tasso di frode](Relazione%20Finale/assets/imbalance_pollution.png)

## Decisioni

**Vettori latenti come feature, non come anomaly score.** Il progetto originale assegna un punteggio alla transazione in base all'errore di ricostruzione. Usare il codice latente come spazio di feature permette a un classificatore supervisionato di sfruttare le etichette già esistenti e rende il confronto fra architetture misurabile con la stessa metrica.

**AE standard come ablation, non PCA.** Il paper confronta l'AAE con metodi classici, non con un autoencoder non regolarizzato di forma identica. Senza quel controllo, una differenza di prestazioni non è attribuibile alla regolarizzazione.

**Stack GPU: RAPIDS cuML al posto di scikit-learn.** Una SVM con kernel RBF su CPU richiedeva ore su 500mila campioni ad alta dimensionalità, rendendo impossibile il tuning; e la t-SNE, con complessità $O(N \log N)$, imponeva campionamenti aggressivi che distruggevano la topologia. `cuml.svm.SVC` e `cuml.manifold.TSNE` hanno portato l'esecuzione a secondi, ordini di grandezza in meno, e hanno reso analizzabile l'intero dataset. È servito un wrapper di compatibilità per mantenere `OneVsRestClassifier` e la pipeline di valutazione esistente.

**Un'immagine Docker fissata invece di un ambiente documentato.** `rapidsai/base:23.10-cuda11.8-py3.9` più PyTorch cu118, così i risultati non dipendono dai driver della macchina ospite. Il prezzo è un'immagine pesante; il guadagno è che un esperimento si ripete identico su un'altra macchina.

## Mappa del repository

```
Encoder.py, Decoder.py, Discriminator.py   le tre reti
validation_V0.py                           addestra l'autoencoder
generateDatasets.py                        materializza i dataset dagli spazi latenti
MAUC.py                                    AUC multi-classe (Hand & Till, 2001)
MAUC_Exp.py                                valutazione con le feature latenti
MAUC_rawExp.py                             valutazione con le 618 feature originali
analyseResults.py                          aggregazione e grafici dei risultati
classification_eval.py                     valutazione lato classificatori
project_extension/                         ablation con AE standard e confronto
Relazione Finale/                          la relazione completa, un capitolo per cartella
Progetto ML.md                             note di lavoro e registro degli esperimenti
```

## Eseguire gli esperimenti

```bash
docker build -t fraud-detection-rapids .
docker run --gpus all -it --rm -v "$PWD":/workspace fraud-detection-rapids
```

Poi, dentro il container, in questo ordine: addestrare l'autoencoder, generare i dataset dagli spazi latenti, ed eseguire le valutazioni (`MAUC_Exp.py` per le feature latenti, `MAUC_rawExp.py` per quelle originali).

**Il dataset non è in questo repository.** Va reperito e messo dove gli script se lo aspettano: è il principale ostacolo a riprodurre i numeri da un clone pulito.

## Limiti

| Cosa | Impatto | Stato |
|---|---|---|
| Il dataset non è versionato | un clone pulito non esegue nulla finché non si fornisce il CSV | aperto, per scelta (dimensione) |
| Nessun test automatico | la pipeline si valida rileggendo i risultati, non con una suite | aperto |
| L'AAE perde contro l'AE standard | la regolarizzazione avversaria non è giustificata su questi dati | misurato, riportato come risultato negativo |
| Crollo oltre il 5% di frodi | il metodo è cieco davanti a un picco di frodi | misurato, richiede monitoraggio del drift in produzione |
| Dati sintetici (PaySim) | i numeri non si trasferiscono a un libro contabile reale senza rivalidare | accettato |

## Documentazione

La relazione completa sta in [`Relazione Finale/`](Relazione%20Finale), un capitolo per cartella:

- **Analisi architetturale e teoria** — il problema dell'anomaly detection finanziaria, autoencoder, adversarial autoencoder
- **Il progetto originale baseline** — il paper di riferimento, il dataset, la pipeline originale, la riproduzione dei risultati
- **Estensione e metodologia** — modernizzazione dello stack, memory engineering della data pipeline, stress test sullo sbilanciamento, stabilità del training, disegno dell'ablation
- **Analisi dei risultati e discussione** — confronto MAUC, analisi t-SNE, stabilità, robustezza all'imbalance
- **Conclusioni e appendici**
