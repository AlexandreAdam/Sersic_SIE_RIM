import tensorflow as tf
import numpy as np
from carrim import PhysicalModel
from carrim.models import CNN
from carrim.utils import nullwriter
import os, time, json
from datetime import datetime


MODEL_HPARAMS = [
    "layers",
    "units",
    "activation",
]

CNN_MODEL_HPARAMS = [
    "levels",
    "layer_per_level",
    "input_kernel_size",
    "filters",
    "activation",
    "architecture"
]


def main(args):

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.json_override is not None:
        if isinstance(args.json_override, list):
            files = args.json_override
        else:
            files = [args.json_override,]
        for file in files:
            with open(file, "r") as f:
                json_override = json.load(f)
            args_dict = vars(args)
            args_dict.update(json_override)

    phys = PhysicalModel(
        pixels=args.pixels,
        image_fov=args.image_fov,
        src_fov=args.src_fov,
        psf_cutout_size=args.psf_cutout_size,
        r_ein_min=args.r_ein_min,
        r_ein_max=args.r_ein_max,
        n_min=args.n_min,
        n_max=args.n_max,
        r_eff_min=args.r_eff_min,
        r_eff_max=args.r_eff_max,
        max_gamma=args.max_gamma,
        max_ellipticity=args.max_ellipticity,
        max_lens_shift=args.max_lens_shift,
        max_source_shift=args.max_source_shift,
        noise_rms_min=args.noise_rms_min,
        noise_rms_max=args.noise_rms_max,
        noise_rms_mean=args.noise_rms_mean,
        noise_rms_std=args.noise_rms_std,
        psf_fwhm_min=args.psf_fwhm_min,
        psf_fwhm_max=args.psf_fwhm_max,
        psf_fwhm_std=args.psf_fwhm_std,
        psf_fwhm_mean=args.psf_fwhm_mean
    )

    cnn_model = CNN(
        architecture=args.cnn_architecture,
        levels=args.cnn_levels,
        layer_per_level=args.cnn_layer_per_level,
        input_kernel_size=args.cnn_input_kernel_size,
        filters=args.cnn_filters,
        activation=args.cnn_activation
    )

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.initial_learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        staircase=args.staircase
    )
    optim = tf.keras.optimizers.deserialize(
        {
            "class_name": args.optimizer,
            'config': {"learning_rate": learning_rate_schedule}
        }
    )

    #   ==== Take care of where to write logs and stuff =================================================================
    if args.model_id.lower() != "none":
        if args.logname is not None:
            logname = args.model_id + "_" + args.logname
            model_id = args.model_id
        else:
            logname = args.model_id + "_" + datetime.now().strftime("%y%m%d%H%M%S")
            model_id = args.model_id
    elif args.logname is not None:
        logname = args.logname
        model_id = logname
    else:
        logname = args.logname_prefixe + "_" + datetime.now().strftime("%y%m%d%H%M%S")
        model_id = logname
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        writer = tf.summary.create_file_writer(logdir)
    else:
        writer = nullwriter()
    # ===== Make sure directory and checkpoint manager are created to save model ===================================
    if args.model_dir.lower() != "none":
        checkpoints_dir = os.path.join(args.model_dir, logname)
        old_checkpoints_dir = os.path.join(args.model_dir, model_id)  # in case they differ we load model from a different directory
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            with open(os.path.join(checkpoints_dir, "script_params.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
            with open(os.path.join(checkpoints_dir, "cnn_model_hparams.json"), "w") as f:
                hparams_dict = {key: vars(args)["cnn_"+key] for key in CNN_MODEL_HPARAMS}
                json.dump(hparams_dict, f, indent=4)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=cnn_model)
        checkpoint_manager = tf.train.CheckpointManager(ckpt, old_checkpoints_dir, max_to_keep=args.max_to_keep)
        save_checkpoint = True
        # ======= Load model if model_id is provided ===============================================================
        if args.model_id.lower() != "none":
            checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint)
        if old_checkpoints_dir != checkpoints_dir:  # save progress in another directory.
            if args.reset_optimizer_states:
                optim = tf.keras.optimizers.deserialize(
                    {
                        "class_name": args.optimizer,
                        'config': {"learning_rate": learning_rate_schedule}
                    }
                )
                ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=cnn_model)
            checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoints_dir, max_to_keep=args.max_to_keep)
    else:
        save_checkpoint = False
    # =================================================================================================================

    # @tf.function
    def train_step(lens, params):
        with tf.GradientTape() as tape:
            tape.watch(cnn_model.trainable_variables)
            y = cnn_model.call(lens)
            cost = tf.reduce_mean(tf.square(y - phys.physical_to_model(params)))
        gradient = tape.gradient(cost, cnn_model.trainable_variables)
        gradient = [tf.clip_by_norm(grad, 5.) for grad in gradient]
        optim.apply_gradients(zip(gradient, cnn_model.trainable_variables))
        return cost

    # ====== Training loop ============================================================================================
    epoch_loss = tf.metrics.Mean()
    time_per_step = tf.metrics.Mean()
    epoch_source_loss = tf.metrics.Mean()
    history = {  # recorded at the end of an epoch only
        "train_cost": [],
        "learning_rate": [],
        "time_per_step": [],
        "step": [],
        "wall_time": []
    }
    best_loss = np.inf
    patience = args.patience
    step = 0
    global_start = time.time()
    estimated_time_for_epoch = 0
    out_of_time = False
    lastest_checkpoint = 1
    for epoch in range(args.epochs):
        if (time.time() - global_start) > args.max_time*3600 - estimated_time_for_epoch:
            break
        epoch_start = time.time()
        epoch_loss.reset_states()
        epoch_source_loss.reset_states()
        time_per_step.reset_states()
        with writer.as_default():
            for batch in range(args.total_items // args.batch_size):
                start = time.time()
                lens, params, noise_rms, psf_fwhm = phys.draw_sersic_batch(args.batch_size)
                cost = train_step(lens, params)
        # ========== Summary and logs ==================================================================================
                _time = time.time() - start
                time_per_step.update_state([_time / args.batch_size])
                epoch_loss.update_state([cost])
                step += 1

            train_cost = epoch_loss.result().numpy()
            tf.summary.scalar("Time per step", time_per_step.result(), step=step)
            tf.summary.scalar("MSE", train_cost, step=step)
            tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
        print(f"epoch {epoch} | train loss {train_cost:.3e} "
              f"| lr {optim.lr(step).numpy():.2e} | time per step {time_per_step.result().numpy():.2e} s")
        history["train_cost"].append(train_cost)
        history["learning_rate"].append(optim.lr(step).numpy())
        history["time_per_step"].append(time_per_step.result().numpy())
        history["step"].append(step)
        history["wall_time"].append(time.time() - global_start)

        cost = train_cost
        if np.isnan(cost):
            print("Training broke the Universe")
            break
        if cost < (1 - args.tolerance) * best_loss:
            best_loss = cost
            patience = args.patience
        else:
            patience -= 1
        if (time.time() - global_start) > args.max_time * 3600:
            out_of_time = True
        if save_checkpoint:
            checkpoint_manager.checkpoint.step.assign_add(1)  # a bit of a hack
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1 or out_of_time:
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    np.savetxt(f, np.array([[lastest_checkpoint, cost]]))
                lastest_checkpoint += 1
                checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint_manager.checkpoint.step),
                                                                checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break
        if out_of_time:
            break
        if epoch > 0:  # First epoch is always very slow and not a good estimate of an epoch time.
            estimated_time_for_epoch = time.time() - epoch_start
        if optim.lr(step).numpy() < 1e-8:
            print("Reached learning rate limit")
            break
    print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
    return history, best_loss


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id",               default="None",                 help="Start from this model id checkpoint. None means start from scratch")

    # Physical parameters
    parser.add_argument("--pixels",                 default=192,    type=int)
    parser.add_argument("--image_fov",              default=7.68,   type=float)
    parser.add_argument("--src_fov",                default=3.,     type=float)
    parser.add_argument("--psf_cutout_size",        default=16,     type=int)
    parser.add_argument("--r_ein_min",              default=0.5,    type=float)
    parser.add_argument("--r_ein_max",              default=2.5,    type=float)
    parser.add_argument("--n_max",                  default=3.,     type=float)
    parser.add_argument("--n_min",                  default=1.,     type=float)
    parser.add_argument("--r_eff_min",              default=0.2,    type=float)
    parser.add_argument("--r_eff_max",              default=1.,     type=float)
    parser.add_argument("--max_gamma",              default=0.05,    type=float)
    parser.add_argument("--max_ellipticity",        default=0.4,    type=float)
    parser.add_argument("--max_lens_shift",         default=0.3,    type=float)
    parser.add_argument("--max_source_shift",       default=0.3,    type=float)
    parser.add_argument("--noise_rms_min",          default=0.01,  type=float)
    parser.add_argument("--noise_rms_max",          default=0.1,    type=float)
    parser.add_argument("--noise_rms_mean",         default=0.05,   type=float)
    parser.add_argument("--noise_rms_std",          default=0.05,    type=float)
    parser.add_argument("--psf_fwhm_min",           default=0.06,   type=float)
    parser.add_argument("--psf_fwhm_max",           default=0.5,    type=float)
    parser.add_argument("--psf_fwhm_mean",          default=0.1,    type=float)
    parser.add_argument("--psf_fwhm_std",           default=0.1,    type=float)

    # CNN model hparams
    parser.add_argument("--cnn_architecture",           default="custom", help="One of ['custom', 'perreault_levasseur2016', 'resnet50', 'resnet50V2', 'resnet101', 'resnet101V2', 'inceptionV3', 'inception_resnetV2']")
    # ... for custom layer only
    parser.add_argument("--cnn_levels",                 default=4,      type=int)
    parser.add_argument("--cnn_layer_per_level",        default=2,      type=int)
    parser.add_argument("--cnn_input_kernel_size",      default=11,      type=int)
    parser.add_argument("--cnn_filters",                default=32,     type=int)
    parser.add_argument("--cnn_activation",             default="relu")

    # Training set params
    parser.add_argument("--batch_size",             default=1,      type=int,       help="Number of images in a batch. ")
    parser.add_argument("--total_items",            default=10,   type=int,       help="Total images in an epoch.")

    # Optimization params
    parser.add_argument("-e", "--epochs",           default=100,     type=int,       help="Number of epochs for training.")
    parser.add_argument("--optimizer",              default="Adamax",               help="Class name of the optimizer (e.g. 'Adam' or 'Adamax')")
    parser.add_argument("--initial_learning_rate",  default=1e-4,   type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",             default=0.5,     type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",            default=1000,    type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",              action="store_true",            help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--patience",               default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",              default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--max_time",               default=np.inf, type=float,     help="Time allowed for the training, in hours.")
    parser.add_argument("--reset_optimizer_states", action="store_true",            help="When training from pre-trained weights, reset states of optimizer.")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",                 default=None,                  help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",         default="CNN",                 help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",               default="None",                help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=1,     type=int,       help="Max model checkpoint to keep.")

    # Reproducibility params
    parser.add_argument("--seed",                   default=None,   type=int,       help="Random seed for numpy and tensorflow.")
    parser.add_argument("--json_override",          default=None,   nargs="+",      help="A json filepath that will override every command line parameters. Useful for reproducibility")

    args = parser.parse_args()

    main(args)
