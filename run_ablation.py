"""Esegue gli studi di ablazione di SplatPose e ne raccoglie i risultati.

Cosa viene misurato
-------------------
Partendo dalla configurazione standard (210 viste di training, 30000 iterazioni
3DGS, k = 175 passi di raffinamento della posa) si cambia un parametro alla
volta:

    viste di training   42, 84, 126, 168  (20, 40, 60, 80% delle 210)
    iterazioni 3DGS     3000, 7000, 15000
    passi della posa k  1, 25, 50, 100, 300

Come si evita di rifare gli stessi calcoli
------------------------------------------
Ogni valutazione è una chiamata a train_and_render.py, lo stesso script della
baseline, con gli argomenti aggiunti per le ablazioni:

  - iterazioni 3DGS: un solo training da 30000 iterazioni salva anche i modelli
    intermedi (-save_iters); ogni modello viene poi valutato con -load_iter;
  - passi k: un solo raffinamento da 300 passi salva il risultato a tutti i
    valori di k (-k_extra); le altre valutazioni rileggono i file (-pose_cache);
  - viste: un training per ogni cartella creata da prepare_view_subsets.py.

Lo script si può interrompere e rilanciare: le valutazioni già complete vengono
riconosciute dal loro summary.json e saltate, e il raffinamento della posa
riprende dall'ultima immagine salvata.

Uso:

    python run_ablation.py -c 02Unicorn 03Mallard 07Owl 16Cat
    python run_ablation.py -c 02Unicorn 03Mallard 07Owl 16Cat -only_collect 1

I modelli finiscono in cartelle `results_ablation_<seed>`, separate da quelle
della baseline, che non vengono toccate. Il log di ogni valutazione va in
`logs/ablation/`, il CSV riassuntivo in `results/`.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

from prepare_view_subsets import prepare_class
from project_paths import project_path

# Configurazione standard e valori alternativi di ciascun parametro.
N_VIEWS = 210
ITERATIONS = 30000
STEPS = 175
VIEWS_VALUES = [42, 84, 126, 168]
ITERATIONS_VALUES = [3000, 7000, 15000]
STEPS_VALUES = [1, 25, 50, 100, 300]

STANDARD_DATA = project_path("MAD-Sim_3dgs")
SUBSETS_DATA = project_path("MAD-Sim_3dgs", "view_subsets")
# Log e tabelle riassuntive stanno nella cartella del progetto, fuori da quella
# dei dati: sono piccoli e servono a chi legge i risultati.
LOG_DIR = project_path("logs", "ablation")
RESULTS_DIR = project_path("results")


def data_path_for(n_views, views_seed):
    """Cartella dati per un certo numero di viste (quella standard se sono tutte)."""
    if n_views == N_VIEWS:
        return STANDARD_DATA
    return os.path.join(SUBSETS_DATA, f"seed{views_seed}", f"views_{n_views:03d}")


def model_dir_for(class_name, n_views, seed, prefix, views_seed):
    """Cartella del modello: stessa regola con cui la costruisce train_and_render.py."""
    return os.path.join(data_path_for(n_views, views_seed), f"results_{prefix}_{seed}", class_name)


def find_evaluations(model_dir, k, iterations):
    """Valutazioni complete già presenti per un modello, a un dato k e checkpoint.

    Restituisce i summary.json trovati, dal più vecchio al più recente. Un
    summary.json esiste solo se la valutazione è arrivata in fondo: viene
    scritto per ultimo da evaluation_export.py.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(model_dir, "evaluation_*", "summary.json"))):
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        config = summary["config"]
        if (summary.get("status") == "complete" and config["k"] == k
                and config.get("load_iter") == iterations):
            found.append(path)
    return found


def evaluate(class_name, seed, prefix, views_seed, n_views=N_VIEWS, iterations=ITERATIONS, k=STEPS):
    """Esegue una valutazione, addestrando il modello se non esiste ancora.

    Non fa nulla se la valutazione è già presente.
    """
    model_dir = model_dir_for(class_name, n_views, seed, prefix, views_seed)
    label = f"{class_name} viste={n_views} iterazioni={iterations} k={k}"
    if find_evaluations(model_dir, k, iterations):
        print(f"[già fatto] {label}")
        return

    command = [sys.executable, "-u", project_path("train_and_render.py"),
               "-c", class_name, "-p", prefix, "-seed", str(seed),
               "-data_path", data_path_for(n_views, views_seed),
               "-gauss_iters", str(ITERATIONS), "-load_iter", str(iterations), "-k", str(k),
               "-pose_cache", "1", "-wandb", "0", "-v", "0", "-save_results", "1"]

    # Il modello esiste se c'è il salvataggio dell'ultima iterazione: è l'ultima
    # cosa che il training scrive, quindi un training interrotto non lo lascia.
    final_model = os.path.join(model_dir, "point_cloud", f"iteration_{ITERATIONS}", "point_cloud.ply")
    trained = os.path.isfile(final_model)
    command += ["-train", "0" if trained else "1"]
    if not trained and n_views == N_VIEWS:
        # Solo il modello standard serve anche a iterazioni intermedie.
        command += ["-save_iters"] + [str(n) for n in ITERATIONS_VALUES]

    if n_views == N_VIEWS and iterations == ITERATIONS:
        # Sul modello standard il raffinamento salva sempre tutti i valori di k,
        # qualunque sia quello valutato in questa chiamata: così anche le
        # immagini elaborate dopo un'interruzione hanno tutti i passi.
        all_steps = sorted(set(STEPS_VALUES) | {STEPS})
        command += ["-k_extra"] + [str(s) for s in all_steps if s != k]

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{class_name}_seed{seed}_views{n_views:03d}_iter{iterations}_k{k}.log")
    print(f"[in corso]  {label}  (log: {log_path})", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=project_path())
    if completed.returncode != 0:
        raise RuntimeError(f"Valutazione fallita ({label}): vedi {log_path}")


def run_class(class_name, seed, prefix, views_seed):
    """Tutte le valutazioni di una classe."""
    # Configurazione standard: addestra il modello (con i salvataggi intermedi)
    # ed esegue l'unico raffinamento da 300 passi.
    evaluate(class_name, seed, prefix, views_seed)
    for k in STEPS_VALUES:
        evaluate(class_name, seed, prefix, views_seed, k=k)
    for iterations in ITERATIONS_VALUES:
        evaluate(class_name, seed, prefix, views_seed, iterations=iterations)
    for n_views in VIEWS_VALUES:
        prepare_class(class_name, n_views, views_seed, STANDARD_DATA, SUBSETS_DATA)
        evaluate(class_name, seed, prefix, views_seed, n_views=n_views)


def mean_pose_time(summary_path):
    """Tempo medio di raffinamento per immagine (ms), letto da scores.csv."""
    with open(os.path.join(os.path.dirname(summary_path), "scores.csv"), newline="", encoding="utf-8") as f:
        times = [float(row["pose_refinement_ms"]) for row in csv.DictReader(f)]
    return sum(times) / len(times)


def collect(classes, seed, prefix, views_seed, output):
    """Raccoglie le metriche di tutte le valutazioni presenti in un file CSV.

    Ogni riga è una valutazione: classe, i tre parametri, le metriche. Le
    configurazioni non ancora eseguite vengono semplicemente saltate.
    """
    configurations = [(N_VIEWS, ITERATIONS, STEPS)]
    configurations += [(n, ITERATIONS, STEPS) for n in VIEWS_VALUES]
    configurations += [(N_VIEWS, n, STEPS) for n in ITERATIONS_VALUES]
    configurations += [(N_VIEWS, ITERATIONS, k) for k in STEPS_VALUES]

    rows = []
    for class_name in classes:
        for n_views, iterations, k in configurations:
            model_dir = model_dir_for(class_name, n_views, seed, prefix, views_seed)
            found = find_evaluations(model_dir, k, iterations)
            if not found:
                continue
            if len(found) > 1:
                print(f"attenzione: {len(found)} valutazioni per {class_name} viste={n_views} "
                      f"iterazioni={iterations} k={k}, uso la più recente")
            with open(found[-1], encoding="utf-8") as f:
                summary = json.load(f)
            metrics = summary["metrics"]
            # La cartella della valutazione è scritta a partire dalla cartella del
            # progetto e con "/": così il CSV vale anche se il progetto viene
            # spostato o copiato su un'altra macchina (anche Linux o Colab) e non
            # contiene il percorso personale di chi lo ha prodotto.
            evaluation_dir = os.path.relpath(os.path.dirname(found[-1]), project_path())
            rows.append({
                "class": class_name, "seed": seed,
                "train_views": n_views, "gs_iterations": iterations, "pose_steps": k,
                "image_auroc": metrics["image_auroc"], "pixel_auroc": metrics["pixel_auroc"],
                "aupro": metrics["aupro"], "pose_ms_per_image": mean_pose_time(found[-1]),
                "evaluation_dir": evaluation_dir.replace(os.sep, "/"),
            })

    if not rows:
        print("Nessuna valutazione trovata.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} valutazioni raccolte in {output}\n")

    # Riepilogo a video: una tabella per parametro, Image AUROC / Pixel AUROC in %.
    def table(title, key, values, fixed):
        print(f"{title}  (Image AUROC / Pixel AUROC, %)")
        print("classe      " + "".join(f"{v:>15}" for v in values))
        for class_name in classes:
            cells = []
            for v in values:
                match = [r for r in rows if r["class"] == class_name and r[key] == v
                         and all(r[other] == value for other, value in fixed.items())]
                cells.append(f"{100 * match[0]['image_auroc']:6.2f} /{100 * match[0]['pixel_auroc']:6.2f}"
                             if match else "-")
            print(f"{class_name:<12}" + "".join(f"{c:>15}" for c in cells))
        print()

    table("Viste di training", "train_views", VIEWS_VALUES + [N_VIEWS],
          {"gs_iterations": ITERATIONS, "pose_steps": STEPS})
    table("Iterazioni 3DGS", "gs_iterations", ITERATIONS_VALUES + [ITERATIONS],
          {"train_views": N_VIEWS, "pose_steps": STEPS})
    table("Passi di raffinamento k", "pose_steps", sorted(STEPS_VALUES + [STEPS]),
          {"train_views": N_VIEWS, "gs_iterations": ITERATIONS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Studi di ablazione di SplatPose")
    parser.add_argument("-c", nargs="+", required=True, metavar="CLASSE",
                        help="una o più classi, per esempio 02Unicorn 16Cat")
    parser.add_argument("-seed", type=int, default=0, help="seed del training e della posa")
    parser.add_argument("-views_seed", type=int, default=0,
                        help="seed con cui prepare_view_subsets.py sceglie le viste")
    parser.add_argument("-p", default="ablation", help="prefisso delle cartelle dei risultati")
    parser.add_argument("-only_collect", type=int, choices=[0, 1], default=0,
                        help="1 = non esegue nulla, raccoglie solo i risultati presenti")
    parser.add_argument("-out", default=None,
                        help="file CSV dei risultati (predefinito: results/ablation_results_seed<seed>.csv)")
    args = parser.parse_args()

    if not args.only_collect:
        for class_name in args.c:
            run_class(class_name, args.seed, args.p, args.views_seed)

    output = args.out or os.path.join(RESULTS_DIR, f"ablation_results_seed{args.seed}.csv")
    collect(args.c, args.seed, args.p, args.views_seed, output)
