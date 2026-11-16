
import json
import os
import random
from project_paths import project_path
import wandb
from evaluation_export import save_evaluation

from gaussian_splatting.train import *

import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.ndimage import gaussian_filter

# needed for PAD code
from easydict import EasyDict
import yaml

classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

pre_parser = ArgumentParser(description="Parameters of the LEGO training run")
pre_parser.add_argument("-k", metavar="K", type=int, help="number of pose estimation steps", default=175)
pre_parser.add_argument("-c", "-classname", metavar="c", type=str, help="current class to run experiments on",
                        default="01Gorilla")
pre_parser.add_argument("-wandb_config", metavar="WC", type=str, help="the wandb config to use", default="None")
pre_parser.add_argument("-p", "-prefix", metavar="pf", type=str, help="prefix for the wandb run name", default="to_delete")
pre_parser.add_argument("-seed", type=int, help="seed for random behavior", default=0)
pre_parser.add_argument("-gauss_iters", type=int, help="number of training iterations for 3DGS", default=30000)
pre_parser.add_argument("-wandb", type=int, help="whether we track with wandb", default=0)
pre_parser.add_argument("-train", type=int, help="whether we train or look for a saved model", default=1)                   
pre_parser.add_argument("-v", type=int, help="verbosity", default=0)                        
pre_parser.add_argument("-save_results", type=int, choices=[0, 1], default=1,
                        help="save scores, metrics, reference images and anomaly maps")
pre_parser.add_argument("-data_path", type=str, help="path pointing towards the usable data set", default=project_path("MAD-Sim_3dgs"))
# Argomenti aggiunti per gli studi di ablazione. Con i valori predefiniti lo
# script si comporta come prima.
pre_parser.add_argument("-load_iter", type=int, default=None,
                        help="3DGS iteration whose saved model is used for pose estimation (default: gauss_iters)")
pre_parser.add_argument("-save_iters", type=int, nargs="*", default=[],
                        help="additional 3DGS iterations at which the model is saved during training")
pre_parser.add_argument("-pose_cache", type=int, choices=[0, 1], default=0,
                        help="save per-image pose estimation results in the model directory and reuse them when present")
pre_parser.add_argument("-k_extra", type=int, nargs="*", default=[],
                        help="additional numbers of pose steps saved in the same run (needs -pose_cache 1)")

lego_args = pre_parser.parse_args()
if lego_args.k_extra and lego_args.pose_cache == 0:
    # Senza salvataggio su disco i risultati agli altri valori di k verrebbero
    # calcolati e poi buttati via: meglio fermarsi subito.
    pre_parser.error("-k_extra needs -pose_cache 1")

data_base_dir = lego_args.data_path

config = {
    "k" : lego_args.k,
    "classname" : lego_args.c,
    "seed" : lego_args.seed,
    "3dgs_iters" : lego_args.gauss_iters,
    "prefix" : lego_args.p,
    "wandb" : lego_args.wandb,
    "train" : lego_args.train,
    "data_dir" : data_base_dir,
    "verbose" : lego_args.v != 0,
    "save_results" : lego_args.save_results != 0,
    # Il codice ufficiale caricava sempre il modello dell'iterazione 30000,
    # anche con -gauss_iters diverso. Ora si carica quello dell'ultima
    # iterazione del training, oppure quello chiesto con -load_iter.
    "load_iter" : lego_args.gauss_iters if lego_args.load_iter is None else lego_args.load_iter,
    "save_iters" : lego_args.save_iters,
    "pose_cache" : lego_args.pose_cache != 0,
    "k_extra" : lego_args.k_extra
}

# Esperimenti con meno viste di training. La cartella dei dati di una classe
# può contenere train_views.json, con gli indici delle viste che ne fanno parte
# (lo scrive prepare_view_subsets.py insieme ai dati). Le stesse viste vanno
# usate per la posa iniziale: una vista che lo splat non ha mai visto non deve
# essere disponibile nemmeno lì. Leggere l'elenco dalla cartella dei dati,
# invece che da un argomento a parte, impedisce che i due vadano fuori sincrono.
# Se il file non c'è (cartella standard) si usano tutte le viste.
views_file = os.path.join(data_base_dir, config["classname"], "train_views.json")
train_views = None
if os.path.isfile(views_file):
    with open(views_file) as f:
        train_views = json.load(f)["indices"]
config["n_train_views"] = None if train_views is None else len(train_views)

projectname = config["prefix"]
if config["wandb"] != 0:
    run = wandb.init(project=projectname, config=config, name=f"{config['prefix']}_{config['classname']}")

data_path = os.path.join(data_base_dir, config["classname"])
result_dir = os.path.join(data_base_dir, f"results_{config['prefix']}_{config['seed']}", config["classname"])
print("saving model to: ", result_dir)
os.makedirs(result_dir, exist_ok=True)

if config["train"] != 0:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    # Set up command line argument parser
    training_args = ["-w", "--eval", "-s", data_path, "-m", result_dir, "--iterations", str(config["3dgs_iters"]), "--sh_degree", "0"]
    if config["save_iters"]:
        # Salvataggi intermedi dello splat. Il programma del training (learning
        # rate, densificazione) non dipende dal numero totale di iterazioni, e
        # a ogni iterazione il salvataggio avviene prima dell'aggiornamento:
        # il modello salvato all'iterazione N di un training lungo è quello che
        # si otterrebbe fermando lo stesso training a N. Un solo training da
        # 30000 iterazioni dà così anche i modelli a 3000, 7000 e 15000.
        # L'iterazione finale viene sempre aggiunta più sotto.
        training_args += ["--save_iterations"] + [str(n) for n in config["save_iters"]]
    print("training args: ", training_args)
    parser = ArgumentParser(description="3DGS Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, config["3dgs_iters"]])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[config["3dgs_iters"]])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(training_args)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, config["seed"])
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    end.record()
    torch.cuda.synchronize()
    train_time_millis = start.elapsed_time(end)
    
    if config["wandb"] != 0:
        wandb.log({
            "train_seconds" : train_time_millis / 1000
        })
        
else:
    if config["wandb"] != 0:
        wandb.log({
            "train_seconds" : 0
        })
    print("skipping training!")
    # Con il training i generatori casuali vengono inizializzati da safe_state.
    # Senza training non lo faceva nessuno, ma il raffinamento della posa parte
    # da parametri casuali (camera_transf): due esecuzioni con lo stesso -seed
    # potevano quindi partire da valori diversi. Qui si fissano gli stessi tre
    # generatori di safe_state, senza toccare il resto (safe_state sostituisce
    # anche sys.stdout, cosa che qui non serve).
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

from pose_estimation import main_pose_estimation
from utils_pose_est import ModelHelper, update_config
from aupro import calculate_au_pro_au_roc


# Con -pose_cache 1 i risultati della posa (un file per immagine di test) stanno
# nella cartella del modello, separati per iterazione del modello 3DGS usato:
# lo stesso modello a iterazioni diverse dà pose diverse.
pose_dir = None
if config["pose_cache"]:
    pose_dir = os.path.join(result_dir, "pose_estimation", f"iteration_{config['load_iter']}")

test_images, reference_images, all_labels, gt_masks, times, image_paths = main_pose_estimation(cur_class=config["classname"],
                                                                                  model_dir_location=result_dir,
                                                                                  k=config["k"], verbose=config["verbose"],
                                                                                  data_dir=None,
                                                                                  return_image_paths=True,
                                                                                  iteration=config["load_iter"],
                                                                                  train_views=train_views,
                                                                                  save_dir=pose_dir,
                                                                                  extra_steps=config["k_extra"])

if config["wandb"] != 0:
    my_data = [[i, times[i]] for i in range(len(times))]
    columns = ["index", "time_millis"]
    cur_table = wandb.Table(data=my_data, columns=columns)
    wandb.log({"time_millis": cur_table})


with open(project_path("PAD_utils", "config_effnet.yaml")) as f:
    mad_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
mad_config = update_config(mad_config)
model = ModelHelper(mad_config.net)
model.eval()
model.cuda()


# evaluation Code taken from PAD/MAD data set paper at https://github.com/EricLee0224/PAD
criterion = torch.nn.MSELoss(reduction='none')
tf_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

tf_mask = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
    ])

test_imgs = list()
score_map_list=list()
scores=list()
pred_list=list()
recon_imgs=list()
with torch.no_grad():
    for i in range(len(test_images)):
        ref=tf_img(reference_images[i]).unsqueeze(0).cuda()
        rgb=tf_img(test_images[i]).unsqueeze(0).cuda()
        ref_feature=model(ref)
        rgb_feature=model(rgb)
        score = criterion(ref, rgb).sum(1, keepdim=True)
        for i in range(len(ref_feature)):
            s_act = ref_feature[i]
            mse_loss = criterion(s_act, rgb_feature[i]).sum(1, keepdim=True)
            score += torch.nn.functional.interpolate(mse_loss, size=224, mode='bilinear', align_corners=False)

        score = score.squeeze(1).cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=4)
        recon_imgs.extend(rgb.cpu().numpy())
        test_imgs.extend(ref.cpu().numpy())
        scores.append(score)

scores = np.asarray(scores).squeeze()
max_anomaly_score = scores.max()
min_anomaly_score = scores.min()
scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
gt_mask = np.concatenate([np.asarray(tf_mask(a))[None,...] for a in gt_masks], axis=0)
precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
a = 2 * precision * recall
b = precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]

fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

au_pro, au_roc, pro_curve, roc_curve = calculate_au_pro_au_roc(gt_mask, scores)
print(f"aupro: {au_pro}. and other au_roc: {au_roc}")

img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list_isano = np.asarray(all_labels) != 0
img_roc_auc = roc_auc_score(gt_list_isano, img_scores)
print('image ROCAUC: %.3f' % (img_roc_auc))

if config["save_results"]:
    export_dir = save_evaluation(
        result_dir, config=config, image_paths=image_paths,
        test_images=test_images, reference_images=reference_images,
        labels=all_labels, masks=gt_mask, scores=scores, pose_times_ms=times,
        metrics={
            "image_auroc": img_roc_auc,
            "pixel_auroc": per_pixel_rocauc,
            "aupro": au_pro,
            "image_auroc_from_masks": au_roc,
        },
        normalization={"raw_min": min_anomaly_score, "raw_max": max_anomaly_score},
    )
    print("Evaluation files saved to:", export_dir)

if config["wandb"] != 0:
    wandb.log({
        "avg_time" : np.mean(times),
        "pixel_roc" : per_pixel_rocauc,
        "image_roc" : img_roc_auc,
        "aupro" : au_pro
    })


