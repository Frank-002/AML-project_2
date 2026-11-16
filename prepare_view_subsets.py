"""Prepara le cartelle dati per gli esperimenti con meno viste di training.

Perché esiste questo file
-------------------------
Uno degli studi di ablazione chiede quanto peggiora SplatPose se lo splat viene
addestrato su meno delle 210 viste disponibili. Il training 3DGS legge le viste
da `transforms_train.json` nella cartella dati della classe: per usarne meno
basta una cartella il cui file elenca solo le viste scelte.

Questo script crea quelle cartelle a partire da quella standard
(MAD-Sim_3dgs/<classe>, prodotta da prepare_lego_ad.py), senza copiare le
immagini: i file JSON puntano alle immagini della cartella standard.

Cosa contiene ogni cartella creata
----------------------------------
    transforms_train.json   solo le viste scelte, con le stesse pose;
    transforms_test.json    come nella cartella standard;
    points3d.ply            copia della nuvola di punti iniziale standard, così
                            ogni esperimento parte dagli stessi punti;
    train_views.json        gli indici delle viste scelte. train_and_render.py
                            lo legge per limitare alle stesse viste anche la
                            posa iniziale con LoFTR.

Come vengono scelte le viste
----------------------------
Si mescolano una volta gli indici 0..209 con un seed fisso; il sottoinsieme da
n viste è formato dai primi n indici mescolati. I sottoinsiemi sono quindi
annidati (le 42 viste sono tutte contenute nelle 84, e così via): passando da
una dimensione alla successiva si aggiungono viste senza toglierne, e le
differenze fra i risultati dipendono solo dalle viste aggiunte.

Uso:

    python prepare_view_subsets.py -c 02Unicorn 03Mallard 07Owl 16Cat -n 42 84 126 168

e poi, per addestrare e valutare su un sottoinsieme:

    python train_and_render.py -c 02Unicorn -data_path MAD-Sim_3dgs/view_subsets/seed0/views_042
"""

import argparse
import json
import os
import shutil

import numpy as np

from project_paths import project_path


def choose_views(n_total, n_views, seed):
    """Indici delle viste di un sottoinsieme, in ordine crescente.

    Con lo stesso seed la mescolata è sempre la stessa, quindi i sottoinsiemi di
    dimensioni diverse sono annidati e la scelta è ripetibile.
    """
    order = np.random.default_rng(seed).permutation(n_total)
    return sorted(int(index) for index in order[:n_views])


def repoint(frames, source_dir, target_dir):
    """Fa puntare i percorsi delle immagini alla cartella dati standard.

    In transforms_*.json `file_path` è relativo alla cartella che contiene il
    file ("./train/train_000"). Nella nuova cartella le immagini non ci sono:
    il percorso viene riscritto come relativo alla nuova cartella ma diretto
    alle immagini originali ("../../../../02Unicorn/train/train_000").
    """
    to_source = os.path.relpath(source_dir, target_dir)
    result = []
    for frame in frames:
        path = os.path.normpath(os.path.join(to_source, frame["file_path"]))
        result.append({**frame, "file_path": path.replace(os.sep, "/")})
    return result


def prepare_class(class_name, n_views, seed, source_base, target_base):
    """Crea la cartella dati di una classe per un sottoinsieme di n_views viste.

    Restituisce il percorso della cartella creata.
    """
    source_dir = os.path.join(source_base, class_name)
    target_dir = os.path.join(target_base, f"seed{seed}", f"views_{n_views:03d}", class_name)

    with open(os.path.join(source_dir, "transforms_train.json")) as f:
        train = json.load(f)
    with open(os.path.join(source_dir, "transforms_test.json")) as f:
        test = json.load(f)

    n_total = len(train["frames"])
    if not 1 <= n_views <= n_total:
        raise ValueError(f"{class_name} ha {n_total} viste: {n_views} non è un numero valido")
    indices = choose_views(n_total, n_views, seed)

    # L'indice j di una vista è la sua posizione in transforms_train.json. È la
    # stessa numerazione delle colonne della tabella LoFTR (coarse_pose.py),
    # perché entrambe seguono l'ordine del transforms.json originale di MAD-Sim.
    views_file = os.path.join(target_dir, "train_views.json")
    if os.path.isfile(views_file):
        with open(views_file) as f:
            existing = json.load(f)["indices"]
        if existing != indices:
            # La cartella può già contenere modelli addestrati su quelle viste:
            # cambiarle renderebbe incoerenti modello e posa iniziale.
            raise ValueError(f"{target_dir} esiste già con un altro insieme di viste")

    os.makedirs(target_dir, exist_ok=True)
    train["frames"] = repoint([train["frames"][j] for j in indices], source_dir, target_dir)
    test["frames"] = repoint(test["frames"], source_dir, target_dir)
    with open(os.path.join(target_dir, "transforms_train.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(target_dir, "transforms_test.json"), "w") as f:
        json.dump(test, f, indent=2)
    with open(views_file, "w") as f:
        json.dump({"indices": indices, "n_total": n_total, "seed": seed}, f)

    # Nuvola di punti iniziale. La cartella standard la contiene dopo il primo
    # training (punti casuali generati dal codice 3DGS). Se non c'è ancora, il
    # training la genererà da solo nella nuova cartella.
    points = os.path.join(source_dir, "points3d.ply")
    if os.path.isfile(points):
        shutil.copyfile(points, os.path.join(target_dir, "points3d.ply"))
    else:
        print(f"  attenzione: {points} non esiste, la nuvola iniziale verrà generata dal training")
    return target_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crea le cartelle dati con un sottoinsieme delle viste di training")
    parser.add_argument("-c", nargs="+", required=True, metavar="CLASSE",
                        help="una o più classi, per esempio 02Unicorn 16Cat")
    parser.add_argument("-n", nargs="+", type=int, required=True, metavar="VISTE",
                        help="numeri di viste da preparare, per esempio 42 84 126 168")
    parser.add_argument("-seed", type=int, default=0,
                        help="seed della scelta delle viste (non è il seed del training)")
    parser.add_argument("-source", default=project_path("MAD-Sim_3dgs"),
                        help="cartella dati standard prodotta da prepare_lego_ad.py")
    parser.add_argument("-target", default=project_path("MAD-Sim_3dgs", "view_subsets"),
                        help="cartella in cui creare i sottoinsiemi")
    args = parser.parse_args()

    for class_name in args.c:
        for n_views in args.n:
            created = prepare_class(class_name, n_views, args.seed, args.source, args.target)
            print(f"{class_name}: {n_views} viste in {created}")
