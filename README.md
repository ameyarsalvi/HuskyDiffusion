# HuskyDiffusion
Diffusion based robot navigation for Skid-steered robots

Director Structrue


├───modules (Contains necessary modules to run the training)
│   │   conv1d.py
│   │   conv_residual.py
│   │   dataset.py
│   │   pose_embedding.py
│   │   resnet.py
│   │   unet.py
│   │   unet2.py
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
│ 
│
└───velDiff
    │   train_v1.py (primary training script)
    │   validate_v1.py (primary valiation script)
    │
    ├───checkpointsV1
    └───runs
