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

    metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in stgcn_metrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)



# from collections import OrderedDict
# from mmaction.models import STGCN
# from mmaction.models import STGCNHead
# import torch
#
# # model = STGCN(
# #         in_channels=3,
# #         edge_importance_weighting=True,
# #         graph_cfg=dict(layout='ntu_edge', strategy='spatial')
# #         )
# model = STGCN(
#     graph_cfg=dict(layout='flag3d', strategy='spatial'),
#     edge_importance_weighting=True,
#     in_channels=3)
#
# modelhead = STGCNHead(
#     num_classes=60,
#     in_channels=256,
#     loss_cls=dict(type='CrossEntropyLoss'))
#
#
# from collections import OrderedDict
# checkpoint = torch.load(
#     '/home/jinpeng/ACTOR/models/actionrecognition/epoch_600.pth')
# new_state_dict = OrderedDict()
# new_state_dict_head = OrderedDict()
# len = len(checkpoint['state_dict'].items())
# for i, (k, v) in enumerate(checkpoint['state_dict'].items()):
#     if i < len-2:
#         if k.startswith('backbone'):
#             name = k[9:]
#             new_state_dict[name] = v
#     else:
#         if k.startswith('cls_head'):
#             name = k[9:]
#             new_state_dict_head[name] = v
#
# model.load_state_dict(new_state_dict)
# modelhead.load_state_dict(new_state_dict_head)
# print(model(torch.rand(10, 3, 16, 24, 1)).size())
# print(modelhead(model(torch.rand(10, 3, 16, 24, 1))).size())


