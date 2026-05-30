import numpy as np

def f0_MCD(peak_freqs, step=0.1, missing_penalty=0.1, update_threshold=0.95):
    
    # min e max considerati
    min_candidate = np.min(peak_freqs) / 2
    max_candidate = np.min(peak_freqs) * 2
    
    # griglia di punti su cui valutare mcd. Parto dall'alto perchè voglio avvantaggiare i multipli piuttosto che i sottomultipli
    candidates = np.arange(max_candidate, min_candidate, -step)

    # inizializza variabili per tenere traccia del miglior candidato
    lowest_residual = float("inf")
    best_f0 = None

    for f0 in candidates:

        # calcola i rapporti tra le frequenze dei picchi e il candidato f0 e arrotonda ai numeri interi più vicini
        ratios = peak_freqs / f0
        integer_ratios = np.round(ratios)
        
        # se uno dei rapporti arrotondati è zero, significa che il candidato f0 è troppo grande e non può essere una fondamentale valida, quindi salto questo candidato
        if np.any(integer_ratios == 0):
            continue

        # calcola i residui tra i rapporti reali e quelli interi e somma i residui totali
        residuals = np.abs(ratios - integer_ratios)
        tot_residual = np.sum(residuals)

        # trova il numero di armoniche mancanti e applica una penalità al residuo totale
        missing_harmonics = np.max(integer_ratios) - len(peak_freqs)
        tot_residual += missing_harmonics * missing_penalty

        # se miglioriamo a sufficienza la stima, aggiorniamo il miglior candidato e il residuo più basso
        if tot_residual < (lowest_residual * update_threshold):
            lowest_residual = tot_residual
            best_f0 = f0
    
    return best_f0
