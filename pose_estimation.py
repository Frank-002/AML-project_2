import os
from pathlib import Path
from project_paths import project_path

from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import DiffGaussianModel
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.render import *
from gaussian_splatting.scene.cameras import Camera

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils_pose_est import DefectDataset, pose_retrieval_loftr, camera_transf
from coarse_pose import get_match_table, select_view

classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

# ---------------------------------------------------------------------------
# Stima della posa per la anomaly detection
# ---------------------------------------------------------------------------
# Per ogni immagine di test:
#   1. posa iniziale: la vista di training con più corrispondenze LoFTR
#      (tabella calcolata una volta sola, vedi coarse_pose.py);
#   2. raffinamento: discesa del gradiente attraverso il renderer (`refine_pose`);
#   3. il rendering alla posa raffinata è il "riferimento" senza difetti con cui
#      l'immagine di test viene confrontata in train_and_render.py.
#
# Il calcolo è quello del codice ufficiale. In più si può:
#   - scegliere il checkpoint 3DGS da caricare (`iteration`);
#   - limitare la posa iniziale a un sottoinsieme di viste (`train_views`);
#   - ottenere il risultato a più valori di k con un solo passaggio (`extra_steps`);
#   - salvare i risultati su disco e rileggerli (`save_dir`).
# Con i valori predefiniti di `main_pose_estimation` il comportamento è quello
# originale.


def checkpoint_path(model_dir, iteration):
    """Percorso dello splat salvato dal training 3DGS a una certa iterazione."""
    return os.path.join(model_dir, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")


def load_render_settings(model_dir):
    """Legge le impostazioni con cui lo splat è stato addestrato.

    Restituisce (dataset, pipeline, background), gli oggetti che il renderer
    3DGS si aspetta. I parametri vengono dal file `cfg_args` che il training
    salva nella cartella del modello; lo sfondo è bianco, come nel training.
    """
    # Set up command line argument parser
    eval_args = ["-w", "--eval", "-m", model_dir]
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser, my_cmdline=eval_args)

    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return dataset, pipeline, background


def refine_pose(test_image, coarse_c2w, splat_path, sh_degree, pipeline, background,
                camera_angle_x, steps):
    """Raffina la posa di un'immagine di test partendo dalla posa iniziale.

    test_image: tensore [3, H, W] in [0, 1], l'immagine di cui si cerca la posa.
    coarse_c2w: matrice 4x4 camera-to-world della vista di training scelta,
        nella convenzione del dataset (transforms.json).
    splat_path: file .ply dello splat addestrato.
    steps: i valori di k a cui mettere da parte il risultato. Il raffinamento
        esegue max(steps) passi.

    Restituisce (snapshots, init_image). `snapshots` ha un elemento per ogni
    valore di `steps`, in ordine crescente di k, con:
        "step"       il valore di k;
        "rendering"  il rendering dello splat a quel punto, [3, H, W] su CPU;
        "R", "T"     la trasformazione appresa che ha prodotto quel rendering;
        "time_ms"    il tempo trascorso dall'inizio, in millisecondi.
    `init_image` è il rendering alla posa iniziale, prima di ogni aggiornamento.

    Come funziona
    -------------
    La camera resta ferma sulla posa iniziale. Si ottimizza invece una
    trasformazione (matrice R, traslazione T) applicata a tutte le gaussiane
    dello splat: spostare l'oggetto davanti a una camera ferma equivale a
    spostare la camera nel verso opposto. `estimated_c2w` ricava da R e T la
    posa della camera equivalente.

    Perché un solo passaggio basta per più valori di k
    --------------------------------------------------
    Il learning rate dipende solo dall'indice del passo, non dal numero totale
    di passi. I primi j passi di un'esecuzione lunga sono quindi gli stessi di
    un'esecuzione che si ferma a j: per avere il risultato a k = 50 e a k = 175
    basta eseguire 175 passi e mettere da parte il rendering lungo la strada.

    Quale rendering corrisponde a "k passi"
    ---------------------------------------
    A ogni iterazione il codice prima disegna lo splat con la posa corrente,
    poi aggiorna la posa. Il codice ufficiale con k passi restituisce il
    rendering dell'ultima iterazione, la numero k-1 contando da 0: è la posa
    dopo k-1 aggiornamenti, perché l'ultimo aggiornamento non viene più
    disegnato. Qui vale la stessa regola: il risultato per k è il rendering
    dell'iterazione k-1. In particolare k = 1 è il rendering alla sola posa
    iniziale di LoFTR, senza raffinamento.
    """
    steps = sorted(set(steps))

    # Il tempo si misura con eventi CUDA, come nel codice ufficiale, e copre le
    # stesse operazioni: conversione della posa, caricamento dello splat, ciclo.
    start = torch.cuda.Event(enable_timing=True)
    start.record()

    # La posa del dataset è camera-to-world con gli assi di Blender/OpenGL (y in
    # alto, la camera guarda lungo -z). Il renderer 3DGS vuole la trasformazione
    # inversa, world-to-camera, con gli assi di COLMAP/OpenCV (y in basso, z in
    # avanti): si cambia segno agli assi y e z, poi si inverte la matrice.
    c2w_init = np.copy(coarse_c2w)
    c2w_init[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w_init)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    c2w_init[:3,:3] = R
    c2w_init[:3,3] = T

    c2w_init = torch.from_numpy(c2w_init).type(torch.float).to("cuda")

    # La trasformazione da ottimizzare parte quasi dall'identità: i suoi
    # parametri sono numeri casuali dell'ordine di 1e-6 (vedi camera_transf).
    # È l'unico punto del raffinamento che usa il generatore casuale.
    cam_transf = camera_transf().to("cuda")
    cam_transf.train()
    optimizer = torch.optim.Adam(cam_transf.parameters(), lr=0.001, betas=(0.9, 0.999))

    # Lo splat viene ricaricato da disco per ogni immagine, come nell'originale.
    gaussians = DiffGaussianModel(sh_degree, c2w_init, cam_transf)
    gaussians.load_ply(splat_path)
    init_image = None

    snapshots = []
    reached = []  # un evento CUDA per ogni snapshot, per misurarne il tempo
    for iters in range(max(steps)):
        optimizer.zero_grad()
        cur_view = Camera(colmap_id=123, R=c2w_init[:3,:3].cpu().numpy(), T=c2w_init[:3,3].cpu().numpy(),
                        FoVx=camera_angle_x, FoVy=camera_angle_x,
                        image=test_image, gt_alpha_mask=None, image_name="aha", uid=123)
        rendering = render(cur_view, gaussians, pipeline, background)["render"]

        if init_image is None:
            init_image = torch.clone(rendering).cpu().detach()

        gt_image = test_image.to("cuda")
        loss = 0.8 * l1_loss(rendering, gt_image) + 0.2 * (1 - ssim(rendering, gt_image))
        loss.backward()

        optimizer.step()

        # Adam parte da 0.001 solo per il primo aggiornamento; da qui in poi il
        # learning rate segue questa curva, che dipende soltanto da `iters`.
        new_lrate = 0.01 * (0.8 ** ((iters + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if iters + 1 in steps:
            # `rendering` è stato disegnato prima dell'aggiornamento appena
            # fatto: è il risultato per k = iters + 1 (vedi la docstring).
            # gaussians.R e gaussians.T sono i valori usati per disegnarlo.
            snapshots.append({
                "step": iters + 1,
                "rendering": rendering.cpu().detach(),
                "R": gaussians.R.detach().cpu().numpy(),
                "T": gaussians.T.detach().cpu().numpy(),
            })
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            reached.append(event)

    torch.cuda.synchronize()
    for snapshot, event in zip(snapshots, reached):
        snapshot["time_ms"] = start.elapsed_time(event)
    return snapshots, init_image


def estimated_c2w(coarse_c2w, R, T):
    """Posa stimata della camera, ricavata dalla trasformazione appresa.

    coarse_c2w: posa iniziale (4x4 camera-to-world, convenzione del dataset).
    R, T: la trasformazione appresa da `refine_pose` (matrice 3x3 e vettore).
    Restituisce una matrice 4x4 camera-to-world nella stessa convenzione di
    transforms.json, quindi confrontabile con le pose di training.

    Da dove viene la formula
    ------------------------
    Il raffinamento lascia la camera sulla posa iniziale e sposta lo splat: ogni
    punto p (vettore riga) diventa  p @ R + T  (DiffGaussianModel.get_xyz).
    Scritto con vettori colonna è  p' = R^T p + T, cioè la matrice 4x4

        M = [[R^T, T], [0, 1]]

    Se W0 è la trasformazione world-to-camera della posa iniziale, un punto
    dello splat spostato finisce in  W0 · M · p. La camera che vedrebbe la stessa
    immagine guardando lo splat originale ha quindi world-to-camera W0 · M, e

        camera-to-world = (W0 · M)^-1 = M^-1 · c2w_iniziale

    Il cambio di assi fra dataset e renderer (segno di y e z) è una
    moltiplicazione a destra della c2w, mentre M^-1 moltiplica a sinistra: le
    due operazioni non si disturbano e la formula vale nella convenzione del
    dataset.

    Nota: R è costruita con la formula di Rodrigues ma l'asse di rotazione non
    viene normalizzato, quindi R è solo approssimativamente una rotazione. Per
    restituire una posa valida la parte di rotazione viene riportata alla
    rotazione più vicina (scomposizione ai valori singolari).
    """
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, dtype=np.float64).T
    M[:3, 3] = np.asarray(T, dtype=np.float64)
    c2w = np.linalg.inv(M) @ np.asarray(coarse_c2w, dtype=np.float64)

    U, _, Vt = np.linalg.svd(c2w[:3, :3])
    c2w[:3, :3] = U @ Vt
    return c2w


def to_uint8(image):
    """Porta un'immagine da valori reali in [0, 1] a interi a 8 bit.

    È la stessa operazione che la valutazione applica come primo passo a ogni
    immagine: `transforms.ToPILImage` in train_and_render.py esegue proprio
    `mul(255).byte()`. Salvare i riferimenti già in questa forma non cambia
    quindi i punteggi, perché è il dato che la valutazione userebbe comunque.
    """
    return image.mul(255).byte()


def from_uint8(image):
    """Inversa di `to_uint8`: da interi a 8 bit a valori reali in [0, 1].

    Il ritorno è esatto: per tutti i 256 valori possibili, applicare di nuovo
    `to_uint8` restituisce l'intero di partenza.
    """
    return image.float() / 255


def result_path(save_dir, index, image_path):
    """File in cui si salva il risultato della posa di un'immagine di test.

    Il nome segue quello usato da evaluation_export.py per la stessa immagine,
    per esempio 0003_Burrs_003.
    """
    image_path = Path(image_path)
    return os.path.join(save_dir, f"{index:04d}_{image_path.parent.name}_{image_path.stem}.npz")


def save_pose_result(path, image_key, coarse_view, coarse_c2w, snapshots):
    """Scrive su disco il risultato della stima della posa di un'immagine.

    Contenuto del file .npz:
        image        nome dell'immagine ("Burrs/003.png"), controllato alla rilettura;
        coarse_view  indice della vista di training usata come posa iniziale;
        coarse_c2w   posa di quella vista (4x4, convenzione del dataset);
        steps        i valori di k salvati, per esempio [1, 25, 50, 100, 175, 300];
    e, con un elemento per ogni valore di `steps`:
        reference    il rendering di riferimento, interi a 8 bit [3, H, W];
        transform_R, transform_T  la trasformazione appresa (vedi `estimated_c2w`);
        time_ms      il tempo impiegato per arrivare a quel passo.

    Si scrive su un file temporaneo e poi lo si rinomina: un'interruzione non
    lascia un file a metà, quindi se il file esiste l'immagine è completa.
    """
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            image=np.array(image_key),
            coarse_view=np.array(coarse_view),
            coarse_c2w=np.asarray(coarse_c2w),
            steps=np.array([s["step"] for s in snapshots]),
            reference=torch.stack([to_uint8(s["rendering"]) for s in snapshots]).numpy(),
            transform_R=np.stack([s["R"] for s in snapshots]),
            transform_T=np.stack([s["T"] for s in snapshots]),
            time_ms=np.array([s["time_ms"] for s in snapshots]),
        )
    os.replace(temporary, path)


def load_pose_result(path, k):
    """Rilegge il risultato salvato per un'immagine, al passo k.

    Restituisce un dizionario con:
        "image"        nome dell'immagine di test;
        "reference"    rendering di riferimento, interi a 8 bit [3, H, W];
        "c2w"          posa stimata della camera (vedi `estimated_c2w`);
        "coarse_view"  indice della vista di training usata come posa iniziale;
        "time_ms"      tempo del raffinamento fino al passo k.
    """
    with np.load(path, allow_pickle=False) as saved:
        steps = saved["steps"].tolist()
        if k not in steps:
            raise ValueError(
                f"{path} contiene i passi {steps} ma non k={k}. I risultati salvati "
                "vengono da un unico passaggio e non si possono estendere: usa uno dei "
                "passi presenti oppure cancella la cartella per ricalcolare.")
        j = steps.index(k)
        return {
            "image": str(saved["image"]),
            "reference": saved["reference"][j],
            "c2w": estimated_c2w(saved["coarse_c2w"], saved["transform_R"][j], saved["transform_T"][j]),
            "coarse_view": int(saved["coarse_view"]),
            "time_ms": float(saved["time_ms"][j]),
        }


def main_pose_estimation(cur_class, model_dir_location, k=150, verbose=False, data_dir=None,
                         return_image_paths=False, iteration=30000, train_views=None,
                         save_dir=None, extra_steps=()):
    """Stima la posa di ogni immagine di test e produce il rendering di riferimento.

    cur_class: nome della classe, per esempio "02Unicorn".
    model_dir_location: cartella del modello 3DGS addestrato.
    k: numero di passi di raffinamento a cui si riferiscono i risultati restituiti.
    verbose: salva immagini di controllo per ogni immagine elaborata.
    data_dir: cartella del dataset MAD-Sim originale (predefinita: MAD-Sim).
    return_image_paths: aggiunge ai risultati i percorsi delle immagini di test.
    iteration: iterazione del training 3DGS di cui caricare lo splat. Il codice
        ufficiale caricava sempre la 30000, che resta il valore predefinito.
    train_views: indici delle viste di training utilizzabili come posa iniziale.
        None = tutte. Negli esperimenti con meno viste vanno passate le stesse
        viste su cui è stato addestrato lo splat.
    save_dir: cartella in cui salvare un file per immagine di test (vedi
        `save_pose_result`). None = non si salva nulla, come nell'originale.
        Le immagini che hanno già il file non vengono rielaborate: si rilegge il
        risultato. Questo permette di riprendere un'esecuzione interrotta e di
        valutare un altro valore di k senza rifare il raffinamento.
    extra_steps: altri valori di k da salvare nello stesso passaggio, oltre a
        `k`. Ha senso solo insieme a `save_dir`.

    Restituisce le liste (immagini di test, rendering di riferimento, etichette,
    maschere, tempi in ms), più i percorsi delle immagini se richiesti. Ogni
    lista segue l'ordine del test set.

    Con `save_dir` i rendering restituiti sono quelli salvati, cioè arrotondati
    a 8 bit. La valutazione applica comunque questo arrotondamento come prima
    operazione (vedi `to_uint8`), quindi i punteggi non cambiano; in cambio una
    seconda esecuzione che rilegge i file valuta esattamente gli stessi dati.

    Un'esecuzione ripresa dopo un'interruzione non è identica bit per bit a una
    senza interruzioni: i numeri casuali con cui parte il raffinamento
    dipendono da quante immagini sono già state elaborate nel processo.
    """
    model_dir = model_dir_location
    data_dir = project_path("MAD-Sim") if data_dir is None else data_dir
    trainset = DefectDataset(data_dir, cur_class, "train", True, True)
    train_poses = np.concatenate([np.array(a["transform_matrix"])[None,...] for a in trainset.camera_transforms["frames"]])

    testset = DefectDataset(data_dir, cur_class, "test", True, True)
    camera_angle_x = trainset.camera_angle

    dataset, pipeline, background = load_render_settings(model_dir)
    splat_path = checkpoint_path(model_dir, iteration)

    # Tutti i valori di k da produrre in questo passaggio.
    steps = sorted(set(extra_steps) | {k})

    # Per ogni immagine di test, il file con il risultato (None se non si salva).
    if save_dir is None:
        saved_files = [None] * len(testset)
    else:
        os.makedirs(save_dir, exist_ok=True)
        saved_files = [result_path(save_dir, i, path) for i, path in enumerate(testset.images)]
    already_saved = [path is not None and os.path.isfile(path) for path in saved_files]

    # Posa iniziale. match_table[i, j] è il numero di corrispondenze LoFTR fra
    # l'immagine di test i e la vista di training j. Viene letta da disco, o
    # calcolata e salvata la prima volta (vedi coarse_pose.py): il codice
    # ufficiale rifaceva questo confronto a ogni esecuzione, dentro il ciclo.
    # Se tutte le immagini hanno già un risultato salvato la tabella non serve.
    match_table = None
    if not all(already_saved):
        match_table = get_match_table(cur_class, trainset, testset)

    if verbose:
        save_to = os.path.join(model_dir_location, "3dgs_imgs")
        os.makedirs(save_to, exist_ok=True)

    normal_images = list()
    reference_images = list()
    all_labels = list()
    gt_masks = list()
    times = list()

    print("STARTING POSE ESTIMATION")

    for i in tqdm(range(len(testset))):
        image_path = Path(testset.images[i])
        filename = f"{image_path.parent.name}_{image_path.stem}.png"
        image_key = f"{image_path.parent.name}/{image_path.name}"

        set_entry = testset[i]

        all_labels.append(set_entry[1])

        gt_masks.append(set_entry[2].cpu().numpy())
        normal_images.append(set_entry[0].cpu().detach())

        if already_saved[i]:
            saved = load_pose_result(saved_files[i], k)
            if saved["image"] != image_key:
                raise ValueError(
                    f"{saved_files[i]} è stato salvato per l'immagine {saved['image']}, "
                    f"non per {image_key}: il test set è cambiato.")
            reference_images.append(from_uint8(torch.from_numpy(saved["reference"])))
            times.append(saved["time_ms"])
            continue

        # La posa di partenza è quella della vista di training più simile
        # all'immagine di test, cioè con più corrispondenze nella tabella.
        coarse_view = select_view(match_table[i], train_views)
        coarse_c2w = train_poses[coarse_view]

        snapshots, init_image = refine_pose(set_entry[0], coarse_c2w, splat_path, dataset.sh_degree,
                                            pipeline, background, camera_angle_x, steps)
        final = snapshots[steps.index(k)]
        rendering = final["rendering"]
        times.append(final["time_ms"])

        if save_dir is None:
            reference_images.append(rendering)
        else:
            save_pose_result(saved_files[i], image_key, coarse_view, coarse_c2w, snapshots)
            # Si restituisce ciò che è stato scritto su disco, non il rendering
            # a piena precisione: così questa esecuzione e quelle che
            # rileggeranno il file valutano lo stesso identico dato.
            reference_images.append(from_uint8(to_uint8(rendering)))

        if verbose:

            cur_save = os.path.join(save_to, Path(filename).stem)
            os.makedirs(cur_save, exist_ok=True)
            torchvision.utils.save_image(set_entry[0].cpu().detach(), os.path.join(cur_save, "gt.png"))
            torchvision.utils.save_image(init_image, os.path.join(cur_save, "first_pose.png"))
            torchvision.utils.save_image(rendering, os.path.join(cur_save, "result.png"))

            fig, axs = plt.subplots(2,2, figsize=(10, 6.4))
            axs[0, 0].set_title("original image"), axs[0, 1].set_title("first pose")
            axs[1,0].set_title(f"iteration {k - 1}"), axs[1,1].set_title(f"diff")
            axs[0,0].imshow(torch.movedim(set_entry[0], 0, 2).cpu().detach())
            axs[0,1].imshow(torch.movedim(init_image, 0, 2))
            axs[1,0].imshow(torch.movedim(rendering, 0, 2))
            axs[1,1].imshow(torch.movedim(torch.abs(rendering - set_entry[0]), 0, 2).sum(dim=2))
            fig.savefig(os.path.join(save_to, filename))
            plt.close(fig)

    print("all labels:", all_labels, len(all_labels))
    for i in range(5):
        print(f"norm: {normal_images[i].shape}. ref: {reference_images[i].shape}. gt: {gt_masks[i].shape}")

    assert len(normal_images) == len(reference_images) == len(testset), f"Wrongly sized sets!" \
                                                                         f"{len(normal_images)}. {len(reference_images)}. {len(testset)}"
    assert len(normal_images) == len(gt_masks), f"Wrongly sized sets! {len(normal_images)}. {len(gt_masks)}"
    result = (normal_images, reference_images, all_labels, gt_masks, times)
    if return_image_paths:
        return (*result, list(testset.images))
    return result


def evaluate_pose_estimation(cur_class, model_dir_location, k=150, verbose=False, data_dir=None):
    
    model_dir = model_dir_location
    
    from utils_pose_est import DatasetPose, pose_retrieval_loftr, camera_transf, matrix_to_quaternion
    import numpy as np
    import wandb

    
    if data_dir is None:
        if cur_class in classnames:
            data_dir = project_path("MAD-Sim_Subsets", "0.8_0", "pose")
        else:
            data_dir = project_path("nerf_synthetic")
    
    trainset = DatasetPose(data_dir, cur_class, "train", True)
    train_imgs = torch.cat([a[0][None,...] for a in trainset], dim=0)
    train_poses = np.concatenate([np.array(a["transform_matrix"])[None,...] for a in trainset.camera_transforms["frames"]])
    train_imgs = torch.movedim(torch.nn.functional.interpolate(train_imgs, (400,400)), 1, 3).numpy()
    
    testset = DatasetPose(data_dir, cur_class, "test", True)    
    
    camera_angle_x = trainset.camera_angle

    all_labels = list()

    # Set up command line argument parser
    eval_args = ["-w", "--eval", "-m", model_dir]
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser, my_cmdline=eval_args)

    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if verbose:
        save_to = os.path.join(model_dir, "img_results")
        os.makedirs(save_to, exist_ok=True)

    translation_errors = list()
    quat_differences = list()
    quat_other1 = list()
    quat_other2 = list()
    coarse_t_error = list()
    coarse_rot_error = list()
    
    print("STARTING POSE ESTIMATION")
    
    for i in tqdm(range(len(testset))):
        
        image_path = Path(testset.images[i])
        filename = f"{image_path.parent.name}_{image_path.stem}.png"
        set_entry = testset[i]
                        
        gt_pose = set_entry[2]
        gt_trans = gt_pose[4:]
        gt_quat = torch.nn.functional.normalize(gt_pose[:4], dim=0)          
        all_labels.append(set_entry[1])
        
        # grab initial pose
        obs_img = torch.movedim(torch.nn.functional.interpolate(set_entry[0][None,...], (400, 400)).squeeze(), 0, 2)
        c2w_init = pose_retrieval_loftr(train_imgs * 255, obs_img.numpy() * 255, train_poses) # my_init_poses[i] # 
        orig_init = torch.from_numpy(np.copy(c2w_init))
        
        # prepare pose for input into gaussian splatting
        c2w_init[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w_init)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        c2w_init[:3,:3] = R
        c2w_init[:3,3] = T
        c2w_init = torch.from_numpy(c2w_init).type(torch.float).to("cuda")
        
        # set up learnable SE(3) transform and splatting model
        cam_transf = camera_transf().to("cuda")
        cam_transf.train()
        gaussians = DiffGaussianModel(dataset.sh_degree, c2w_init, cam_transf)
        gaussians.load_ply(os.path.join(model_dir,
                                        "point_cloud",
                                        "iteration_" + str(30000),
                                        "point_cloud.ply"))
        optimizer = torch.optim.Adam(cam_transf.parameters(), lr=0.001, betas=(0.9, 0.999))
        init_image = None
        
        # calculate errors of coarse pose estimation
        coarse_t_error.append(torch.sqrt(torch.sum((gt_trans - orig_init[:3,3]) ** 2)).item())
        init_quat = torch.nn.functional.normalize(matrix_to_quaternion(orig_init[:3,:3]), dim=0).type(torch.float)
        coarse_rot_error.append(torch.arccos(torch.abs(torch.dot(gt_quat, init_quat))).item())        
           
        for iters in range(k):
            optimizer.zero_grad()
            cur_view = Camera(colmap_id=123, R=c2w_init[:3,:3].cpu().numpy(), T=c2w_init[:3,3].cpu().numpy(),
                            FoVx=camera_angle_x, FoVy=camera_angle_x,
                            image=set_entry[0], gt_alpha_mask=None, image_name="aha", uid=123)
            rendering = render(cur_view, gaussians, pipeline, background)["render"]


            if init_image is None:
                init_image = torch.clone(rendering).cpu().detach()

            gt_image = set_entry[0].to("cuda")
            
            loss = 0.8 * l1_loss(rendering, gt_image) + 0.2 * (1 - ssim(rendering, gt_image))
            loss.backward()
            optimizer.step()
            
            new_lrate = 0.01 * (0.8 ** ((iters + 1) / 100))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
                
                
            if iters == k - 1:
                # construct estimated SE(3) transform
                estimation = torch.zeros((4,4), dtype=torch.float32)
                estimation[:3,:3] = gaussians.R
                estimation[:3,3] = gaussians.T
                estimation[3,3] = 1.0
                                
                # apply my estimated SE(3) transform to the original initial pose 
                est_t = estimation.detach() @ orig_init.type(torch.float32)
                trans_error = torch.sqrt(torch.sum((gt_trans - est_t[:3,3]) ** 2))
                
                est_quat = torch.nn.functional.normalize(matrix_to_quaternion(est_t[:3,:3]), dim=0)
                
                # measured as ph_1 from "Metrics for 3D Rotations: Comparison and Analysis"
                # which ranges from [0,sqrt(2)]
                norm_of_diff = min(torch.linalg.vector_norm(gt_quat - est_quat, ord=2),
                                   torch.linalg.vector_norm(gt_quat + est_quat, ord=2))
                # phi_2, this one is used for the SplatPose paper                
                quat_diff = torch.arccos(min(torch.abs(torch.dot(gt_quat, est_quat)), torch.tensor(1.0)))
                # phi_3
                quat_diff_2 = 1 - torch.abs(torch.dot(gt_quat, est_quat))
                
                translation_errors.append(trans_error.item())
                quat_differences.append(norm_of_diff)
                quat_other1.append(quat_diff)
                quat_other2.append(quat_diff_2)
                                
                
                if verbose:
                    cur_save = os.path.join(save_to, Path(filename).stem)
                    os.makedirs(cur_save, exist_ok=True)
                    torchvision.utils.save_image(set_entry[0].cpu().detach(), os.path.join(cur_save, "gt.png"))
                    torchvision.utils.save_image(init_image, os.path.join(cur_save, "first_pose.png"))
                    torchvision.utils.save_image(rendering.cpu().detach(), os.path.join(cur_save, "result.png"))
                    continue
                    
    ret_dict = {
        "translation_error" : translation_errors,
        "rotation_error" : quat_differences,
        "rotation_error_other1" : quat_other1,
        "rotation_error_other2" : quat_other2,
        "coarse_t_error" : coarse_t_error,
        "coarse_rot_error" : coarse_rot_error
    }
    
    my_data = [[testset.images[i], coarse_t_error[i], coarse_rot_error[i], translation_errors[i], quat_differences[i], quat_other1[i], quat_other2[i]] for i in range(len(testset))]
    columns = ["path", "coarse_t", "coarse_rot", "trans", "rot_1", "rot_2", "rot_3"]
    cur_table = wandb.Table(data=my_data, columns=columns)
    # wandb.log({"pose_est": cur_table})
    
    assert all([len(a) == len(testset) for a in ret_dict.values()]), "Wrongly sized lists of errors!"
    return ret_dict
    