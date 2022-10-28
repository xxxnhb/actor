import torch
from tqdm import tqdm

from src.utils.fixseed import fixseed
from src.evaluate.action2motion.evaluate import A2MEvaluation_flag3d
from src.evaluate.stgcn.evaluate import Evaluation as STGCNEvaluation
# from src.evaluate.othermetrics.evaluation import Evaluation
from mmaction.models import STGCN
from torch.utils.data import DataLoader
from src.utils.tensors import collate

import os

from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model
from src.datasets.get_dataset import get_datasets
import src.utils.rotation_conversions as geometry
from src.evaluate.action2motion.evaluate import A2MEvaluation


def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x


class NewDataloader:
    def __init__(self, mode, model, parameters, dataiterator, device):
        assert mode in ["gen", "rc", "gt"]

        pose_rep = parameters["pose_rep"]
        translation = parameters["translation"]

        self.batches = []

        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    classes = databatch["y"]
                    gendurations = databatch["lengths"]
                    batch = model.generate(classes, gendurations)
                    feats = "output"
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    feats = "x"
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    feats = "output"

                batch = {key: val.to(device) for key, val in batch.items()}

                if translation:
                    x = batch[feats][:, :-1]
                else:
                    x = batch[feats]

                x = x.permute(0, 3, 1, 2)
                x = convert_x_to_rot6d(x, pose_rep)
                x = x.permute(0, 2, 3, 1)

                batch["x"] = x

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)


class NewDataloader_flag3d:
    def __init__(self, mode, model, dataiterator, device):
        assert mode in ["gen", "rc", "gt"]
        self.batches = []
        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    classes = databatch["y"]
                    gendurations = databatch["lengths"]
                    batch = model.generate(classes, gendurations)
                    batch = {key: val.to(device) for key, val in batch.items()}
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    batch["x_xyz"] = model.rot2xyz(batch["x"].to(device),
                                                   batch["mask"].to(device))
                    batch["output"] = batch["x"]
                    batch["output_xyz"] = batch["x_xyz"]
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    batch["output_xyz"] = model.rot2xyz(batch["output"],
                                                        batch["mask"])
                    batch["x_xyz"] = model.rot2xyz(batch["x"],
                                                   batch["mask"])

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)


def evaluate(parameters, folder, checkpointname, epoch, niter):
    torch.multiprocessing.set_sharing_strategy('file_system')

    bs = parameters["batch_size"]
    doing_recons = False

    device = parameters["device"]
    dataname = parameters["dataset"]

    # dummy => update parameters info
    # get_datasets(parameters)
    # faster: hardcode value for uestc

    parameters["num_classes"] = 40
    parameters["nfeats"] = 6
    parameters["njoints"] = 25

    model = get_gen_model(parameters)
    print(model)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    model.outputxyz = False

    recogparameters = parameters.copy()
    recogparameters["pose_rep"] = "rot6d"
    recogparameters["nfeats"] = 6

    # Action2motionEvaluation
    stgcnevaluation = STGCNEvaluation(dataname, recogparameters, device)

    stgcn_metrics = {}
    # joints_metrics = {}
    # pose_metrics = {}

    compute_gt_gt = False
    if compute_gt_gt:
        datasetGT = {key: [get_datasets(parameters)[key],
                           get_datasets(parameters)[key]]
                     for key in ["train", "test"]}
    else:
        datasetGT = {key: [get_datasets(parameters)[key]]
                     for key in ["train", "test"]}

    print("Dataset loaded")

    allseeds = list(range(niter))

    for seed in allseeds:
        fixseed(seed)
        for key in ["train", "test"]:
            for data in datasetGT[key]:
                data.reset_shuffle()
                data.shuffle()

        dataiterator = {key: [DataLoader(data, batch_size=bs,
                                         shuffle=False, num_workers=8,
                                         collate_fn=collate)
                              for data in datasetGT[key]]
                        for key in ["train", "test"]}

        if doing_recons:
            reconsLoaders = {key: NewDataloader("rc", model, parameters,
                                                dataiterator[key][0],
                                                device)
                             for key in ["train", "test"]}

        gtLoaders = {key: NewDataloader("gt", model, parameters,
                                        dataiterator[key][0],
                                        device)
                     for key in ["train", "test"]}

        if compute_gt_gt:
            gtLoaders2 = {key: NewDataloader("gt", model, parameters,
                                             dataiterator[key][1],
                                             device)
                          for key in ["train", "test"]}

        genLoaders = {key: NewDataloader("gen", model, parameters,
                                         dataiterator[key][0],
                                         device)
                      for key in ["train", "test"]}

        loaders = {"gen": genLoaders,
                   "gt": gtLoaders}
        if doing_recons:
            loaders["recons"] = reconsLoaders

        if compute_gt_gt:
            loaders["gt2"] = gtLoaders2

        stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders)
        del loaders

        # joints_metrics = evaluation.evaluate(model, loaders, xyz=True)
        # pose_metrics = evaluation.evaluate(model, loaders, xyz=False)

    metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in
                         stgcn_metrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)


def evaluate_flag3d(parameters, folder, checkpointname, epoch, niter):
    num_frames = 60
    bs = parameters["batch_size"]
    doing_recons = False

    parameters["num_frames"] = num_frames
    parameters["jointstype"] = "smpl"
    parameters["vertstrans"] = True
    parameters["nfeats"] = 3
    parameters["num_classes"] = 60

    device = parameters["device"]
    dataname = parameters["dataset"]

    # dummy => update parameters info
    get_datasets(parameters)
    model = get_gen_model(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.outputxyz = True

    a2mevaluation = A2MEvaluation_flag3d(dataname, device, parameters)
    a2mmetrics = {}

    # evaluation = OtherMetricsEvaluation(device)
    # joints_metrics = {}, pose_metrics = {}

    datasetGT1 = get_datasets(parameters)["train"]
    datasetGT2 = get_datasets(parameters)["train"]

    allseeds = list(range(niter))

    try:
        for index, seed in enumerate(allseeds):
            print(f"Evaluation number: {index + 1}/{niter}")
            fixseed(seed)

            datasetGT1.reset_shuffle()
            datasetGT1.shuffle()

            datasetGT2.reset_shuffle()
            datasetGT2.shuffle()

            dataiterator = DataLoader(datasetGT1, batch_size=parameters["batch_size"],
                                      shuffle=False, num_workers=8, collate_fn=collate)
            dataiterator2 = DataLoader(datasetGT2, batch_size=parameters["batch_size"],
                                       shuffle=False, num_workers=8, collate_fn=collate)

            # reconstructedloader = NewDataloader("rc", model, dataiterator, device)
            print("device is ", device)
            motionloader = NewDataloader_flag3d("gen", model, dataiterator, device)
            gt_motionloader = NewDataloader_flag3d("gt", model, dataiterator, device)
            gt_motionloader2 = NewDataloader_flag3d("gt", model, dataiterator2, device)

            # Action2motionEvaluation
            loaders = {"gen": motionloader,
                       # "recons": reconstructedloader,
                       "gt": gt_motionloader,
                       "gt2": gt_motionloader2}

            a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)

            # joints_metrics[seed] = evaluation.evaluate(model, num_classes,
            # loaders, xyz=True)
            # pose_metrics[seed] = evaluation.evaluate(model, num_classes,
            # loaders, xyz=False)

    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in
                         a2mmetrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
