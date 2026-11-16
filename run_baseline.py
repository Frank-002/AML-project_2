"""Esegue la baseline di SplatPose su più classi e più seed e ne raccoglie i risultati.

Cosa viene misurato
-------------------
La configurazione standard (210 viste di training, 30000 iterazioni 3DGS,
k = 175 passi di raffinamento della posa) ripetuta con seed diversi. Ogni run è
un nuovo training dello splat seguito dalla valutazione completa: la media sui
seed è il valore da confrontare con il paper, la deviazione standard dice
quanto i risultati cambiano da un'esecuzione all'altra.

Come funziona
-------------
Ogni run è una chiamata a train_and_render.py con gli stessi argomenti dei run
originali della baseline. Questo script non calcola nulla da sé: decide quali
run mancano, li lancia uno alla volta e alla fine legge i loro summary.json.

Lo script si può interrompere e rilanciare:

  - un run che ha già un summary.json completo viene saltato;
  - se lo splat è già addestrato ma la valutazione manca, viene rifatta solo
    la valutazione (-train 0): lo splat non viene riaddestrato né sovrascritto.

Uso:

    python run_baseline.py -c 02Unicorn 03Mallard 07Owl 16Cat
    python run_baseline.py -c 02Unicorn 03Mallard 07Owl 16Cat -only_collect 1

I modelli finiscono in `MAD-Sim_3dgs/results_<prefisso>_<seed>/<classe>`, il
log di ogni run in `logs/baseline/`, il CSV riassuntivo in `results/`.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

from project_paths import project_path
# La configurazione standard è definita una volta sola, in run_ablation.py.
# Importarla da lì garantisce che la baseline e il punto standard della griglia
# di ablazione usino gli stessi valori.
from run_ablation import ITERATIONS, RESULTS_DIR, STANDARD_DATA, STEPS, mean_pose_time

SEEDS = [0, 1, 2, 3, 4]
LOG_DIR = project_path("logs", "baseline")
# Metriche riportate, con il nome che hanno in summary.json e quello stampato.
METRICS = {"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC", "aupro": "AUPRO"}


def model_dir_for(class_name, seed, prefix):
    """Cartella del modello: stessa regola con cui la costruisce train_and_render.py."""
    return os.path.join(STANDARD_DATA, f"results_{prefix}_{seed}", class_name)


def find_baseline_evaluations(model_dir):
    """Valutazioni complete della configurazione standard presenti per un modello.

    Restituisce i summary.json trovati, dal più vecchio al più recente. Un
    summary.json esiste solo se la valutazione è arrivata in fondo: viene
    scritto per ultimo da evaluation_export.py.

    Perché non si usa find_evaluations di run_ablation.py: quella funzione
    richiede la chiave `load_iter` nel riepilogo. I run della baseline eseguiti
    prima degli studi di ablazione non la contengono, perché allora il codice
    caricava sempre lo splat dell'iterazione 30000. Qui la chiave assente vale
    quindi ITERATIONS. Senza questa regola quei run sembrerebbero mancanti e
    verrebbero rifatti tutti.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(model_dir, "evaluation_*", "summary.json"))):
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        config = summary["config"]
        if (summary.get("status") == "complete" and config["k"] == STEPS
                and config["3dgs_iters"] == ITERATIONS
                and config.get("load_iter", ITERATIONS) == ITERATIONS):
            found.append(path)
    return found


def run(class_name, seed, prefix):
    """Esegue un run della baseline. Non fa nulla se il run è già completo."""
    model_dir = model_dir_for(class_name, seed, prefix)
    label = f"{class_name} seed={seed}"
    if find_baseline_evaluations(model_dir):
        print(f"[già fatto] {label}")
        return

    # Lo splat esiste se c'è il salvataggio dell'ultima iterazione: è l'ultima
    # cosa che il training scrive, quindi un training interrotto non lo lascia.
    # Se c'è, manca solo la valutazione: -train 0 la esegue senza riaddestrare.
    final_model = os.path.join(model_dir, "point_cloud", f"iteration_{ITERATIONS}", "point_cloud.ply")
    trained = os.path.isfile(final_model)

    # Gli stessi argomenti dei run originali. -pose_cache resta spento, come
    # allora: i rendering di riferimento vanno alla valutazione a piena
    # precisione e non vengono scritti file di posa.
    command = [sys.executable, "-u", project_path("train_and_render.py"),
               "-c", class_name, "-p", prefix, "-seed", str(seed),
               "-data_path", STANDARD_DATA,
               "-gauss_iters", str(ITERATIONS), "-k", str(STEPS),
               "-train", "0" if trained else "1",
               "-wandb", "0", "-v", "0", "-save_results", "1"]

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{class_name}_seed{seed}.log")
    print(f"[in corso]  {label}  (log: {log_path})", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=project_path())
    if completed.returncode != 0:
        raise RuntimeError(f"Run fallito ({label}): vedi {log_path}")


def collect(classes, seeds, prefix, output):
    """Raccoglie le metriche dei run presenti in un file CSV e stampa il riepilogo.

    Ogni riga del CSV è un run: classe, seed, metriche a piena precisione (fra
    0 e 1, come in summary.json). I run non ancora eseguiti vengono saltati.
    """
    rows = []
    for class_name in classes:
        for seed in seeds:
            found = find_baseline_evaluations(model_dir_for(class_name, seed, prefix))
            if not found:
                continue
            if len(found) > 1:
                print(f"attenzione: {len(found)} valutazioni per {class_name} seed={seed}, uso la più recente")
            with open(found[-1], encoding="utf-8") as f:
                metrics = json.load(f)["metrics"]
            # Percorso scritto a partire dalla cartella del progetto e con "/",
            # come in run_ablation.py: il CSV non contiene il percorso personale
            # di chi lo ha prodotto e vale anche su un'altra macchina.
            evaluation_dir = os.path.relpath(os.path.dirname(found[-1]), project_path())
            rows.append({
                "class": class_name, "seed": seed,
                **{name: metrics[name] for name in METRICS},
                "pose_ms_per_image": mean_pose_time(found[-1]),
                "evaluation_dir": evaluation_dir.replace(os.sep, "/"),
            })

    if not rows:
        print("Nessun run trovato.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} run raccolti in {output}\n")

    # Riepilogo a video, in percentuale: per ogni classe media ± deviazione
    # standard sui seed. È la deviazione standard descrittiva dei run (divisore
    # = numero di run, `pstdev`), non un intervallo di confidenza.
    print("Media ± deviazione standard sui seed (%)")
    print(f"{'classe':<12}{'run':>4}" + "".join(f"{title:>18}" for title in METRICS.values()))
    class_means = []
    for class_name in classes:
        runs = [row for row in rows if row["class"] == class_name]
        if not runs:
            continue
        means, cells = [], []
        for name in METRICS:
            values = [100 * row[name] for row in runs]
            means.append(statistics.mean(values))
            cells.append(f"{means[-1]:.2f} ± {statistics.pstdev(values):.2f}")
        class_means.append(means)
        print(f"{class_name:<12}{len(runs):>4}" + "".join(f"{cell:>18}" for cell in cells))
    # Media sulle classi: ogni classe pesa allo stesso modo, qualunque sia il
    # numero dei suoi run. I sette spazi finali allineano il valore sotto le
    # medie delle righe sopra (lo spazio occupato da " ± 0.00").
    overall = [statistics.mean(column) for column in zip(*class_means)]
    print(f"{'Media':<12}{'':>4}" + "".join(f"{value:>11.2f}{'':7}" for value in overall))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline di SplatPose su più classi e più seed")
    parser.add_argument("-c", nargs="+", required=True, metavar="CLASSE",
                        help="una o più classi, per esempio 02Unicorn 16Cat")
    parser.add_argument("-seeds", nargs="+", type=int, default=SEEDS, metavar="SEED",
                        help="seed da eseguire (predefiniti: 0 1 2 3 4)")
    parser.add_argument("-p", default="baseline_windows",
                        help="prefisso delle cartelle dei risultati (predefinito: quello dei run originali)")
    parser.add_argument("-only_collect", type=int, choices=[0, 1], default=0,
                        help="1 = non esegue nulla, raccoglie solo i risultati presenti")
    parser.add_argument("-out", default=None,
                        help="file CSV dei risultati (predefinito: results/baseline_results.csv)")
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))

    if not args.only_collect:
        for class_name in args.c:
            # Seed in ordine crescente, come nei run originali. Il primo
            # training di una classe genera la nuvola di punti iniziale
            # (points3d.ply nella cartella dati della classe) e tutti i
            # training successivi la riusano: cambiare l'ordine cambierebbe la
            # nuvola da cui partono tutti i seed di quella classe.
            for seed in seeds:
                run(class_name, seed, args.p)

    output = args.out or os.path.join(RESULTS_DIR, "baseline_results.csv")
    collect(args.c, seeds, args.p, output)
