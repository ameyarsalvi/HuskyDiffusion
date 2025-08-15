# HuskyDiffusion
Diffusion based robot navigation for Skid-steered robots

Directory Structrue (updated 8/9/2025)

```
├───modules (Contains necessary modules to run the training)
│   │   conv1d.py
│   │   conv_residual.py
│   │   dataset.py
│   │   pose_embedding.py
│   │   resnet.py
│   │   unetX.py
│   │   datasetvX.py
│
├───UnitTesting
│   │   test_conv1d.py
│   │   test_dataset.py
│   │   test_mydataset.py
│   │   test_myresnet.py
│   │   test_pose_embedding.py
│   │   test_residual_block.py
│   │   test_resnet.py
│   │   test_unet1d.py
│   │   visualize_pose_embedding.py
│   │
│   ├───moduleUnderstanding
│   │       understandConv1D.ipynb
│   │       understandDatasetLoader.ipynb
│   │       understandResidual1D.ipynb
│   │       understandUNet1D.ipynb
|
├───velDiff
|    │   train_vX.py (primary training script)
|    │   validate_vX.py (primary valiation script)
|    ├───conf
|    |      |   config.yaml
|    ├───checkpointsV1
|    └───runs (try to empty unless necessary)
|
├───dataset_utils
|    │   cmd_vel_heatmap.py
|    |   merge_csvs.py
|    |   mod_dataset.py
|    |   move_images.py
|
└───plot_utils
    |   validate_plot_viz.py
    |   viz_preds.py
    |   viz_spline.py

```

