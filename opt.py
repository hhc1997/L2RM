import os
import time
import argparse
def get_options():
    current_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

    # Data Settings
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument(
        "--data_path", default="./data", help="path to datasets"
    )
    parser.add_argument(
        "--data_name", default="f30k_precomp", help="{coco,f30k,cc152k}_precomp"
    )
    parser.add_argument(
        "--vocab_path",
        default="./vocab",
        help="Path to saved vocabulary json files.",
    )

    # Training Settings
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Size of a training mini-batch."
    )
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr_update",
        default=15,
        type=int,
        help="Number of epochs to update the learning rate.",
    )
    parser.add_argument(
        "--lr", default=2e-4, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="Number of data loader workers."
    )
    parser.add_argument(
        "--log_step",
        default=1000,
        type=int,
        help="Number of steps to print and record the log.",
    )
    parser.add_argument(
        "--grad_clip", default=2.0, type=float, help="Gradient clipping threshold."
    )
    parser.add_argument("--margin", default=0.2, type=float, help="Rank loss margin.")

    # Model Settings
    parser.add_argument(
        "--img_dim",
        default=2048,
        type=int,
        help="Dimensionality of the image embedding.",
    )
    parser.add_argument(
        "--word_dim",
        default=300,
        type=int,
        help="Dimensionality of the word embedding.",
    )
    parser.add_argument(
        "--embed_size",
        default=1024,
        type=int,
        help="Dimensionality of the joint embedding.",
    )
    parser.add_argument(
        "--sim_dim", default=256, type=int, help="Dimensionality of the sim embedding."
    )
    parser.add_argument(
        "--num_layers", default=1, type=int, help="Number of GRU layers."
    )
    parser.add_argument("--bi_gru", action="store_false", help="Use bidirectional GRU.")
    parser.add_argument(
        "--no_imgnorm",
        action="store_true",
        help="Do not normalize the image embeddings.",
    )
    parser.add_argument(
        "--no_txtnorm",
        action="store_true",
        help="Do not normalize the text embeddings.",
    )
    parser.add_argument("--module_name", default="SGR", type=str, help="SGR, SAF")
    parser.add_argument("--sgr_step", default=3, type=int, help="Step of the SGR.")

    # Noise(PMPs) Settings
    parser.add_argument("--noise_file", default="", help="noise_file")
    parser.add_argument("--noise_ratio", default=0.2, type=float, help="Noisy ratio")

    # Warmup Settings
    parser.add_argument("--warmup_epoch", default=5, type=int, help="warm up epochs")
    parser.add_argument("--warmup_model_path", default="", help="warm up models")

    # Runing Settings
    parser.add_argument("--seed", default=0, type=int, help="init the seed")

    parser.add_argument("--gpu", default="0", help="Which gpu to use.")

    parser.add_argument(
        "--output_dir", default=os.path.join("output", current_time), help="Output dir."
    )
    parser.add_argument("--comment", default="", type=str, help="some important comments")

    # OT Settings
    parser.add_argument("--queue_length", default=128, type=int, help="queue number of images.")
    parser.add_argument(
        "--lr_cost", default=2e-6, type=float, help="Initial learning rate of the cost function."
    )
    parser.add_argument("--tau", default=0.05, type=float, help="temperature coefficient")
    parser.add_argument("--reg", default=0.07, type=float, help="Sinkhorn regularization parameter")
    parser.add_argument("--rho", default=0.1, type=float, help="partial transport mass")


    return parser.parse_args()