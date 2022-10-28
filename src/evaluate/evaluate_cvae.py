from src.parser.evaluation import parser


def main():
    parameters, folder, checkpointname, epoch, niter = parser()

    dataset = parameters["dataset"]
    print(dataset)
    if dataset in ["ntu13", "humanact12"]:
        from .gru_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    elif dataset in ["uestc"]:
        from .stgcn_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    elif dataset in ["flag3d", "flag3d_r003"]:
        from .stgcn_eval import evaluate_flag3d
        evaluate_flag3d(parameters, folder, checkpointname, epoch, niter)
    else:
        raise NotImplementedError("This dataset is not supported.")


if __name__ == '__main__':
    main()
