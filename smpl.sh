python -m src.generate.generate_sequences $1checkpoint_$2.pth.tar --num_samples_per_action 10 --cpu
python -m src.render.rendermotion "$1"generation.npy
