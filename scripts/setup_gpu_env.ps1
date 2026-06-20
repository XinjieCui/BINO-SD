param(
    [string]$EnvName = "pde_inverse_operator_gpu",
    [string]$TorchVersion = "2.7.1",
    [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$envList = conda env list
if (-not ($envList | Select-String -Pattern "^\s*$EnvName\s")) {
    conda create -n $EnvName python=3.11 pip numpy scipy matplotlib -c conda-forge -y
}

conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install "torch==$TorchVersion" --index-url $CudaIndexUrl
conda run -n $EnvName python -m pip install -e .

Write-Host "GPU environment ready: $EnvName"
Write-Host "Validation command:"
Write-Host "conda run -n $EnvName python -c `"import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)`""
