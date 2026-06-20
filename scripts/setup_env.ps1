param(
    [string]$EnvName = "pde_inverse_operator"
)

$envList = conda env list
if (-not ($envList | Select-String -Pattern "^\s*$EnvName\s")) {
    conda env create -f ".\environment.yml"
}

conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run -n $EnvName python -m pip install -e .
