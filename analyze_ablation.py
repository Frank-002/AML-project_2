"""Diagnostica degli studi di ablazione: perché le configurazioni danno risultati diversi.

Perché esiste questo file
-------------------------
run_ablation.py dice QUANTO cambiano le metriche al variare di viste, iterazioni
3DGS e passi di raffinamento. Questo script misura, sugli stessi file già
salvati, le due cose che possono spiegare PERCHÉ cambiano:

  1. la posa: l'immagine di test è stata allineata al modello oppure no?
  2. la ricostruzione: lo splat riproduce bene l'oggetto, anche da viste che
     non ha usato nel training?

Non riesegue né training né raffinamento: legge i file di posa
(pose_estimation/iteration_<N>/*.npz) e i punteggi (scores.csv) prodotti dalla
griglia. Solo la misura della ricostruzione (-render 1) usa la GPU, per
disegnare alcune viste di training con ciascun modello.

Come si giudica la posa senza conoscere quella vera
---------------------------------------------------
Le immagini di test di MAD-Sim non hanno una posa nota. Come riferimento si usa
la posa stimata dal modello standard (210 viste, 30000 iterazioni) dopo il
numero massimo di passi salvato (300): è la stima migliore disponibile. Tutti i
modelli sono addestrati con le stesse pose di training, quindi stanno nello
stesso sistema di coordinate e le loro pose sono confrontabili.

Un'immagine è considerata "allineata" se l'angolo tra l'orientamento finale
della camera e quello di riferimento è al massimo ALIGNED_DEGREES (2°).
La posizione della camera può differire anche quando il criterio è soddisfatto.

Limite da tenere presente: il riferimento è a sua volta una stima. Dove 300
passi non bastano (alcune immagini di Cat) il riferimento stesso può essere
impreciso, e a k = 300 tutte le immagini risultano allineate per definizione.

Cosa viene scritto
------------------
Un file CSV in `results/`, con una riga per configurazione e classe. Le colonne
principali:

    start_angle_median     distanza fra la vista iniziale scelta da LoFTR e il
                           riferimento: quanto è lontano il punto di partenza;
    start_wrong_view       immagini la cui vista iniziale dista più di
                           WRONG_VIEW_DEGREES: LoFTR ha scelto una vista che non
                           è fra le più vicine (spesso il lato opposto);
    unaligned              immagini la cui posa finale non è allineata;
    unaligned_good         quante di queste sono immagini sane. Contano molto:
                           un'immagine sana disallineata riceve un punteggio
                           alto e abbassa l'Image AUROC;
    image_auroc_aligned    Image AUROC calcolata solo sulle immagini allineate
                           sia in questa configurazione sia in quella standard;
    standard_auroc_same_images
                           Image AUROC della configurazione standard sulle
                           stesse immagini. Eventuali differenze di AUROC possono
                           ancora risentire di scarti di posizione e orientamento;
    auroc_burrs / auroc_missing / auroc_stains
                           Image AUROC fra le immagini sane e un solo tipo di
                           difetto;
    psnr_seen / psnr_unseen
                           (solo con -render 1) qualità della ricostruzione su
                           due gruppi fissi di 42 viste di training: quelle
                           comuni a tutti i sottoinsiemi e quelle escluse da
                           tutti i sottoinsiemi. Per i modelli con meno viste
                           il secondo gruppo è fatto di viste mai usate.

Uso:

    python analyze_ablation.py -c 02Unicorn 03Mallard 07Owl 16Cat -render 1
"""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import PureWindowsPath

import numpy as np
from sklearn.metrics import roc_auc_score

from pose_estimation import checkpoint_path, estimated_c2w, load_render_settings
from run_ablation import (ITERATIONS, ITERATIONS_VALUES, N_VIEWS, RESULTS_DIR, STANDARD_DATA, STEPS,
                          STEPS_VALUES, VIEWS_VALUES, data_path_for, find_evaluations, model_dir_for)

ALIGNED_DEGREES = 2.0      # massimo scarto angolare ammesso rispetto al riferimento
WRONG_VIEW_DEGREES = 45.0  # oltre questa distanza la vista iniziale non è una delle vicine
REFERENCE_STEPS = max(STEPS_VALUES + [STEPS])
DEFECT_TYPES = ("Burrs", "Missing", "Stains")


def rotation_angle(pose_a, pose_b):
    """Angolo in gradi fra le orientazioni di due pose camera-mondo 4x4."""
    cosine = (np.trace(pose_a[:3, :3].T @ pose_b[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def load_poses(model_dir, iteration):
    """Pose salvate per un modello: vista iniziale e posa a ogni passo salvato.

    Restituisce {immagine: {"start": posa della vista LoFTR,
                            "steps": {k: posa stimata dopo k passi}}}.
    Dai file si leggono solo le matrici, non i rendering (sono grandi e qui non
    servono).
    """
    poses = {}
    pattern = os.path.join(model_dir, "pose_estimation", f"iteration_{iteration}", "*.npz")
    for path in sorted(glob.glob(pattern)):
        with np.load(path) as data:
            start = data["coarse_c2w"]
            steps = [int(step) for step in data["steps"]]
            rotations, translations = data["transform_R"], data["transform_T"]
            image = str(data["image"])
        poses[image] = {
            "start": start,
            "steps": {step: estimated_c2w(start, rotations[i], translations[i]) for i, step in enumerate(steps)},
        }
    return poses


def load_scores(summary_path):
    """Punteggio, etichetta e categoria di ogni immagine di una valutazione.

    La chiave è "categoria/file", la stessa scritta nei file di posa.
    """
    scores = {}
    path = os.path.join(os.path.dirname(summary_path), "scores.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = f"{row['category']}/{PureWindowsPath(row['image_path']).name}"
            scores[image] = {"score": float(row["image_score"]),
                             "anomalous": int(row["is_anomaly_dataset"]),
                             "category": row["category"]}
    return scores


def image_auroc(scores, images):
    """Image AUROC su un sottoinsieme di immagini (None se manca una delle due classi)."""
    labels = [scores[image]["anomalous"] for image in images]
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, [scores[image]["score"] for image in images]))


def gaussian_count(model_dir, iteration):
    """Numero di gaussiane di un modello, letto dall'intestazione del file PLY."""
    with open(checkpoint_path(model_dir, iteration), "rb") as f:
        header = f.read(2000).decode("ascii", errors="ignore")
    return int(re.search(r"element vertex (\d+)", header).group(1))


def camera_distances(class_name, reference):
    """Distanza dall'origine delle camere di training e delle pose di test stimate.

    Le camere di training di MAD-Sim stanno tutte alla stessa distanza
    dall'origine. Se le pose di test stimate stanno a un'altra distanza, il
    raffinamento deve anche avvicinare o allontanare la camera, non solo
    ruotarla: è uno spostamento in più da compiere.
    """
    with open(os.path.join(STANDARD_DATA, class_name, "transforms_train.json")) as f:
        frames = json.load(f)["frames"]
    train = np.array([np.linalg.norm(np.array(frame["transform_matrix"])[:3, 3]) for frame in frames])
    test = np.array([np.linalg.norm(pose[:3, 3]) for pose in reference.values()])
    return float(np.median(train)), float(np.median(test)), float(test.min()), float(test.max())


def fixed_view_groups(class_name, views_seed):
    """Due gruppi fissi di viste di training su cui misurare la ricostruzione.

    seen:   le viste del sottoinsieme più piccolo. I sottoinsiemi sono annidati,
            quindi ogni modello le ha usate nel training.
    unseen: le viste fuori dal sottoinsieme più grande, quindi mai usate da
            nessun modello con meno viste (quello con 210 viste le ha usate).
    """
    def indices(n_views):
        with open(os.path.join(data_path_for(n_views, views_seed), class_name, "train_views.json")) as f:
            return json.load(f)["indices"]

    seen = indices(min(VIEWS_VALUES))
    largest = set(indices(max(VIEWS_VALUES)))
    unseen = [j for j in range(N_VIEWS) if j not in largest]
    return seen, unseen


def reconstruction_psnr(class_name, model_dir, iteration, view_groups):
    """PSNR medio fra le viste di training vere e le stesse viste disegnate dal modello.

    Le immagini e le camere vengono caricate con le stesse funzioni del training
    3DGS, e il PSNR è calcolato come nel suo resoconto: i valori sono
    confrontabili con quelli stampati nei log del training.
    """
    # Import qui dentro: servono solo con -render 1 e richiedono la GPU.
    import torch
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene import GaussianModel
    from gaussian_splatting.scene.dataset_readers import readCamerasFromTransforms
    from gaussian_splatting.utils.camera_utils import loadCam
    from gaussian_splatting.utils.image_utils import psnr

    dataset, pipeline, background = load_render_settings(model_dir)
    # Percorso assoluto: il lettore 3DGS unisce due volte la cartella al nome del file.
    source = os.path.abspath(os.path.join(STANDARD_DATA, class_name))
    infos = readCamerasFromTransforms(source, "transforms_train.json", dataset.white_background)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(checkpoint_path(model_dir, iteration))

    results = []
    with torch.no_grad():
        for group in view_groups:
            values = []
            for j in group:
                camera = loadCam(dataset, j, infos[j], 1.0)
                image = torch.clamp(render(camera, gaussians, pipeline, background)["render"], 0.0, 1.0)
                truth = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
                values.append(psnr(image, truth).mean().item())
            results.append(float(np.mean(values)))
    return results


def diagnose_class(class_name, seed, prefix, views_seed, with_render):
    """Tutte le misure per una classe: una riga per configurazione."""
    standard_dir = model_dir_for(class_name, N_VIEWS, seed, prefix, views_seed)
    standard_poses = load_poses(standard_dir, ITERATIONS)
    reference = {image: entry["steps"][REFERENCE_STEPS] for image, entry in standard_poses.items()}
    images = sorted(reference)
    good = [image for image in images if image.startswith("good/")]

    # Immagini allineate nella configurazione standard: servono per confrontare
    # ogni altra configurazione con lo standard sulle stesse immagini.
    standard_scores = load_scores(find_evaluations(standard_dir, STEPS, ITERATIONS)[-1])
    standard_aligned = {image for image in images
                        if rotation_angle(standard_poses[image]["steps"][STEPS], reference[image]) <= ALIGNED_DEGREES}

    train_distance, test_distance, test_min, test_max = camera_distances(class_name, reference)
    print(f"\n===== {class_name}: {len(images)} immagini di test, {len(good)} sane")
    print(f"distanza delle camere dall'origine: training {train_distance:.2f}, "
          f"pose di test stimate {test_distance:.2f} (da {test_min:.2f} a {test_max:.2f})")

    # Quanto è vicino il punto di partenza, per categoria: spiega i punteggi
    # senza raffinamento (k = 1), che misurano soprattutto il disallineamento.
    by_category = {}
    for image in images:
        angle = rotation_angle(standard_poses[image]["start"], reference[image])
        by_category.setdefault(image.split("/")[0], []).append(angle)
    print("vista iniziale entro 5 gradi dal riferimento, per categoria: "
          + ", ".join(f"{category} {sum(a <= 5 for a in angles)}/{len(angles)}"
                      for category, angles in sorted(by_category.items())))

    view_groups = fixed_view_groups(class_name, views_seed) if with_render else None

    configurations = [(N_VIEWS, ITERATIONS, STEPS)]
    configurations += [(n, ITERATIONS, STEPS) for n in VIEWS_VALUES]
    configurations += [(N_VIEWS, n, STEPS) for n in ITERATIONS_VALUES]
    configurations += [(N_VIEWS, ITERATIONS, k) for k in STEPS_VALUES]

    rows = []
    psnr_cache = {}  # i valori di k condividono il modello: si disegna una volta sola
    for n_views, iterations, k in configurations:
        model_dir = model_dir_for(class_name, n_views, seed, prefix, views_seed)
        found = find_evaluations(model_dir, k, iterations)
        if not found:
            continue
        is_standard_model = n_views == N_VIEWS and iterations == ITERATIONS
        poses = standard_poses if is_standard_model else load_poses(model_dir, iterations)
        scores = load_scores(found[-1])

        start = np.array([rotation_angle(poses[image]["start"], reference[image]) for image in images])
        final = {image: rotation_angle(poses[image]["steps"][k], reference[image]) for image in images}
        unaligned = [image for image in images if final[image] > ALIGNED_DEGREES]
        common = [image for image in images if final[image] <= ALIGNED_DEGREES and image in standard_aligned]

        row = {
            "class": class_name, "seed": seed,
            "train_views": n_views, "gs_iterations": iterations, "pose_steps": k,
            "images": len(images), "good_images": len(good),
            "camera_distance_train": train_distance, "camera_distance_test_median": test_distance,
            "start_angle_median": float(np.median(start)),
            "start_angle_p90": float(np.percentile(start, 90)),
            "start_wrong_view": int((start > WRONG_VIEW_DEGREES).sum()),
            "final_angle_median": float(np.median(list(final.values()))),
            "unaligned": len(unaligned),
            "unaligned_good": sum(image.startswith("good/") for image in unaligned),
            "image_auroc": image_auroc(scores, images),
            "aligned_in_common": len(common),
            "aligned_in_common_good": sum(image.startswith("good/") for image in common),
            "image_auroc_aligned": image_auroc(scores, common),
            "standard_auroc_same_images": image_auroc(standard_scores, common),
        }
        for defect in DEFECT_TYPES:
            subset = [image for image in images if scores[image]["category"] in ("good", defect)]
            row[f"auroc_{defect.lower()}"] = image_auroc(scores, subset)
        row["gaussians"] = gaussian_count(model_dir, iterations)

        row["psnr_seen"], row["psnr_unseen"] = None, None
        if with_render:
            key = (model_dir, iterations)
            if key not in psnr_cache:
                psnr_cache[key] = reconstruction_psnr(class_name, model_dir, iterations, view_groups)
            row["psnr_seen"], row["psnr_unseen"] = psnr_cache[key]
        rows.append(row)

        def percent(value):
            return "   -  " if value is None else f"{100 * value:6.2f}"

        print(f"viste {n_views:>3} iter {iterations:>5} k {k:>3} | partenza {row['start_angle_median']:5.1f} gradi, "
              f"viste sbagliate {row['start_wrong_view']:>3} | non allineate {len(unaligned):>3} (sane {row['unaligned_good']:>2}) | "
              f"AUROC {percent(row['image_auroc'])}, sulle allineate {percent(row['image_auroc_aligned'])} "
              f"contro {percent(row['standard_auroc_same_images'])} dello standard (n={len(common)})"
              + (f" | PSNR {row['psnr_seen']:.1f} / {row['psnr_unseen']:.1f}" if with_render else ""))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostica degli studi di ablazione di SplatPose")
    parser.add_argument("-c", nargs="+", required=True, metavar="CLASSE",
                        help="una o più classi, per esempio 02Unicorn 16Cat")
    parser.add_argument("-seed", type=int, default=0, help="seed della griglia da analizzare")
    parser.add_argument("-views_seed", type=int, default=0, help="seed dei sottoinsiemi di viste")
    parser.add_argument("-p", default="ablation", help="prefisso delle cartelle dei risultati")
    parser.add_argument("-render", type=int, choices=[0, 1], default=0,
                        help="1 = misura anche la qualità della ricostruzione (usa la GPU)")
    parser.add_argument("-out", default=None,
                        help="file CSV della diagnostica (predefinito: results/ablation_diagnostics_seed<seed>.csv)")
    args = parser.parse_args()

    output = args.out or os.path.join(RESULTS_DIR, f"ablation_diagnostics_seed{args.seed}.csv")

    # Protezione: senza -render 1 le colonne del PSNR restano vuote. Se il file
    # di destinazione contiene già quei valori, riscriverlo li cancellerebbe e
    # per riaverli bisognerebbe rifare i rendering. Meglio fermarsi subito.
    if not args.render and os.path.isfile(output):
        with open(output, newline="", encoding="utf-8") as f:
            has_psnr = any(row.get("psnr_seen") for row in csv.DictReader(f))
        if has_psnr:
            parser.error(f"{output} contiene già i valori di PSNR e questa esecuzione non li calcola: "
                         "usare -render 1 per ricalcolarli oppure -out per scrivere in un altro file")

    rows = []
    for class_name in args.c:
        rows += diagnose_class(class_name, args.seed, args.p, args.views_seed, bool(args.render))

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} righe scritte in {output}")
