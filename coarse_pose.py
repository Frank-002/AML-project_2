"""Posa iniziale ("coarse") con LoFTR, con le corrispondenze salvate su disco.

Perché esiste questo file
-------------------------
Per ogni immagine di test SplatPose sceglie una posa di partenza: confronta
l'immagine con TUTTE le viste di training tramite LoFTR e prende la posa della
vista con più corrispondenze. Quella posa viene poi raffinata dal renderer.

Nel codice ufficiale (`utils_pose_est.pose_retrieval_loftr`) questo confronto
viene rifatto a ogni esecuzione e LoFTR viene ricaricato da disco per ogni
immagine. Ma il risultato dipende solo dalle immagini del dataset: non dipende
dallo splat, dal numero di iterazioni 3DGS né dai passi di raffinamento. Quindi
lo calcoliamo UNA volta per classe e lo salviamo come tabella

    counts[i, j] = numero di corrispondenze LoFTR fra l'immagine di test i
                   e la vista di training j

Tutti gli esperimenti successivi leggono la tabella invece di rieseguire LoFTR.
La tabella conserva i conteggi verso tutte le viste, non solo la vincitrice:
così, quando un esperimento usa un sottoinsieme di viste di training, basta
cercare il massimo fra le colonne ammesse (vedi `select_view`).

Cosa NON cambia rispetto al codice ufficiale
--------------------------------------------
Le immagini passano per le stesse identiche operazioni (ridimensionamento a
400x400, scala di grigi, valori in [0, 1]) e la vista scelta è la stessa: quella
con più corrispondenze, e a parità la prima in ordine di indice.

Uso da riga di comando, per preparare le tabelle in anticipo:

    python coarse_pose.py -c 02Unicorn 03Mallard 07Owl 16Cat
"""

import argparse
import os
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from PAD_utils.loftr import LoFTR, default_cfg
from project_paths import project_path
from utils_pose_est import DefectDataset

# Lato delle immagini date a LoFTR. È il valore del codice ufficiale: il
# raffinamento della posa lavora invece a 800x800.
LOFTR_SIZE = 400

# Valore delle celle non ancora calcolate. Un conteggio vero è sempre >= 0,
# quindi -1 distingue "non calcolato" da "zero corrispondenze".
NOT_COMPUTED = -1

# Ogni quante immagini di test la tabella parziale viene scritta su disco.
# Un'interruzione costa al massimo questo numero di immagini.
SAVE_EVERY = 20


def load_matcher():
    """Costruisce LoFTR con i pesi preaddestrati, pronto per l'inferenza su GPU.

    Stessa configurazione e stessi pesi del codice ufficiale. La differenza è
    che qui il modello si carica una volta sola, non una volta per immagine.
    """
    config = deepcopy(default_cfg)
    config["coarse"]["temp_bug_fix"] = True  # richiesto dal checkpoint "ds_new"
    matcher = LoFTR(config=config)
    checkpoint = torch.load(project_path("PAD_utils", "model", "indoor_ds_new.ckpt"))
    matcher.load_state_dict(checkpoint["state_dict"])
    return matcher.eval().cuda()


def to_loftr_gray(image):
    """Prepara un'immagine per LoFTR.

    image: tensore [3, H, W] con valori in [0, 1], come restituito da DefectDataset.
    Restituisce un tensore [1, 1, 400, 400] in scala di grigi su GPU.

    Le operazioni sono le stesse del codice ufficiale, nello stesso ordine, per
    ottenere esattamente gli stessi valori:
      1. ridimensionamento a 400x400 (interpolate usa "nearest" di default);
      2. passaggio a [0, 255] e conversione in grigio con OpenCV;
      3. ritorno a [0, 1].
    """
    small = torch.nn.functional.interpolate(image[None, ...], (LOFTR_SIZE, LOFTR_SIZE))
    rgb = torch.movedim(small.squeeze(0), 0, 2).numpy() * 255  # [400, 400, 3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return torch.from_numpy(gray)[None][None].cuda() / 255.


def count_matches(matcher, query, gallery):
    """Conta le corrispondenze LoFTR fra un'immagine e ogni vista della galleria.

    query: tensore [1, 1, 400, 400] (da `to_loftr_gray`).
    gallery: lista di tensori dello stesso formato, una per vista di training.
    Restituisce un array di interi lungo quanto la galleria.

    LoFTR restituisce una confidenza per ogni corrispondenza trovata (`mconf`):
    il numero di corrispondenze è la lunghezza di quel vettore, come nel codice
    ufficiale.
    """
    counts = np.empty(len(gallery), dtype=np.int32)
    with torch.no_grad():
        for j, view in enumerate(gallery):
            batch = {"image0": query, "image1": view}
            matcher(batch)  # LoFTR scrive i risultati dentro `batch`
            counts[j] = len(batch["mconf"])
    return counts


def select_view(counts_row, allowed=None):
    """Sceglie la vista di training da cui parte il raffinamento della posa.

    counts_row: la riga della tabella relativa a un'immagine di test.
    allowed: indici delle viste di training utilizzabili. None = tutte.
        Serve per gli esperimenti con meno viste: le viste escluse dallo splat
        non devono essere disponibili nemmeno per la posa iniziale.
    Restituisce l'indice della vista, riferito alla lista completa delle viste.

    A parità di corrispondenze vince l'indice più basso. È lo stesso
    comportamento del ciclo ufficiale, che aggiorna il massimo solo con un
    confronto stretto (`>`): `np.argmax` restituisce la prima occorrenza.
    """
    if allowed is None:
        candidates = np.arange(len(counts_row))
    else:
        # Ordine crescente: così "la prima" fra le ammesse è quella che il
        # ciclo ufficiale incontrerebbe per prima su una galleria ridotta.
        candidates = np.sort(np.asarray(allowed))
    if np.any(counts_row[candidates] == NOT_COMPUTED):
        raise ValueError("La tabella LoFTR non è completa per questa immagine")
    return int(candidates[np.argmax(counts_row[candidates])])


def cache_path(cur_class, cache_dir=None):
    """Percorso del file con la tabella di una classe.

    La cartella predefinita sta in MAD-Sim_3dgs insieme agli altri file
    generati, ma fuori dalle cartelle `results_*`: la tabella non appartiene a
    un modello o a un seed, vale per tutti gli esperimenti sulla classe.
    """
    if cache_dir is None:
        cache_dir = project_path("MAD-Sim_3dgs", "loftr_cache")
    return os.path.join(cache_dir, f"{cur_class}.npz")


def _save_table(path, counts, test_keys, train_keys):
    """Scrive la tabella su disco senza rischiare un file a metà.

    Si scrive prima un file temporaneo e poi lo si rinomina: la rinomina è
    un'operazione unica, quindi un'interruzione lascia intatta la versione
    precedente invece di un file corrotto.
    """
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        np.savez(handle, counts=counts,
                 test_images=np.array(test_keys), train_images=np.array(train_keys))
    os.replace(temporary, path)


def get_match_table(cur_class, trainset, testset, cache_dir=None):
    """Restituisce la tabella delle corrispondenze, calcolando ciò che manca.

    trainset, testset: i DefectDataset della classe (viste di training e
        immagini di test), gli stessi usati dalla pose estimation.
    Restituisce un array [n_test, n_train] di interi; la riga i segue l'ordine
    di `testset`, la colonna j l'ordine di `trainset`.

    Se il file esiste ed è completo non viene eseguito nulla su GPU. Se è
    parziale (esecuzione interrotta) si calcolano solo le righe mancanti.
    """
    path = cache_path(cur_class, cache_dir)

    # Nomi delle immagini, salvati insieme ai conteggi. Servono a controllare
    # che la tabella su disco descriva proprio queste immagini in quest'ordine:
    # un ordine diverso assocerebbe in silenzio i conteggi all'immagine sbagliata.
    test_keys = [f"{Path(p).parent.name}/{Path(p).name}" for p in testset.images]
    train_keys = [Path(p).name for p in trainset.images]

    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as saved:
            counts = saved["counts"]
            same_images = (saved["test_images"].tolist() == test_keys
                           and saved["train_images"].tolist() == train_keys)
        if not same_images:
            raise ValueError(
                f"La tabella LoFTR in {path} è stata calcolata su immagini diverse "
                "da quelle attuali. Cancella il file per ricalcolarla.")
    else:
        counts = np.full((len(test_keys), len(train_keys)), NOT_COMPUTED, dtype=np.int32)

    missing = np.flatnonzero((counts == NOT_COMPUTED).any(axis=1))
    if len(missing) == 0:
        return counts

    os.makedirs(os.path.dirname(path), exist_ok=True)
    matcher = load_matcher()
    # Le viste di training si preparano una volta e restano su GPU: sono le
    # stesse per tutte le immagini di test (210 viste occupano circa 130 MB).
    gallery = [to_loftr_gray(trainset[j][0]) for j in range(len(trainset))]

    for done, i in enumerate(tqdm(missing, desc=f"LoFTR {cur_class}"), start=1):
        counts[i] = count_matches(matcher, to_loftr_gray(testset[i][0]), gallery)
        if done % SAVE_EVERY == 0:
            _save_table(path, counts, test_keys, train_keys)
    _save_table(path, counts, test_keys, train_keys)

    # LoFTR non serve più: si libera la memoria GPU per il raffinamento.
    del matcher, gallery
    torch.cuda.empty_cache()
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcola e salva le tabelle delle corrispondenze LoFTR")
    parser.add_argument("-c", nargs="+", required=True, metavar="CLASSE",
                        help="una o più classi, per esempio 02Unicorn 16Cat")
    parser.add_argument("-data_path", default=project_path("MAD-Sim"),
                        help="cartella del dataset MAD-Sim originale")
    parser.add_argument("-cache_dir", default=None,
                        help="cartella delle tabelle (predefinita: MAD-Sim_3dgs/loftr_cache)")
    args = parser.parse_args()

    for class_name in args.c:
        train = DefectDataset(args.data_path, class_name, "train", True, True)
        test = DefectDataset(args.data_path, class_name, "test", True, True)
        table = get_match_table(class_name, train, test, args.cache_dir)
        print(f"{class_name}: tabella {table.shape[0]} x {table.shape[1]} "
              f"in {cache_path(class_name, args.cache_dir)}")
