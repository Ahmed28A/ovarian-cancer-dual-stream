# Weights

The trained Stream A checkpoint (ConvNeXt-Base + CBAM, about 350 MB) is too large for a normal GitHub file.

1. Get `best_convnext_single_cbam.pth` from the training run.
   Notebook 01 saves it to `MyDrive/Ovarian_GradProject/checkpoints/`.
2. Rename it to `stream_a_convnext_single_cbam.pth` and put it in this folder.
3. To deploy, upload the file to the Hugging Face Space, which stores large files with Git LFS.
   You can also attach it to a GitHub Release.

The clinical model (Stream B) needs no weights file, because it is trained in a few seconds when the app starts.
