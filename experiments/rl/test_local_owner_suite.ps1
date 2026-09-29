param(
    [string]$Python = 'C:/Users/Administrator/miniconda3/envs/pytorch/python.exe',
    [string]$Python312 = 'C:/Users/Administrator/miniconda3/python.exe'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$receipts = Join-Path $repo 'research/temporary/rl_upstream_alignment_20260929/local-environment-final'
$sources = Join-Path $repo 'research/temporary/rl_upstream_alignment_20260929/recipe-sources'
$deps = Join-Path $repo 'research/temporary/rl_local_test_deps'
$verl = Join-Path $sources 'verl-agent-20bd331'
$sky = Join-Path $sources 'SkyRL-7d94ccf0eac3439c1731ce32018bf043dd639806'
$agent = Join-Path $sources 'AgentGym-d014732d9fe39b975c368c03749bfd50950067f6/agentenv'
$previousPythonPath = $env:PYTHONPATH

function Invoke-OwnerTests([string]$Name, [string[]]$Paths, [string[]]$Tests, [bool]$Modern = $false) {
    $testPython = if ($Modern) { $Python312 } else { $Python }
    $testDeps = if ($Modern) { @((Join-Path $repo 'research/temporary/rl_local_model_entry_deps'), (Join-Path $repo 'research/temporary/rl_local_test_deps312')) } else { @($deps) }
    $env:PYTHONPATH = (@($testDeps) + @($PSScriptRoot) + $Paths) -join ';'
    & $testPython (Join-Path $PSScriptRoot 'run_local_cpu_tests.py') `
        --receipt (Join-Path $receipts "$Name.json") -- $testPython -m pytest -q @Tests `
        "--junitxml=$receipts/$Name.xml"
    if ($LASTEXITCODE -ne 0) { throw "$Name failed; see $receipts/$Name.log" }
}

Push-Location $repo
try {
    # Environment unit tests only. No algorithm, DT, model or service runs.
    # Each framework gets its own process to keep its native import graph.
    Invoke-OwnerTests 'recipes' @($sky, "$sky/skyrl-gym") @('experiments/rl/test_official_recipes_local.py')
    Invoke-OwnerTests 'sql-owner' @($sky, "$sky/skyrl-gym") @("$sky/skyrl-gym/tests/test_sql.py")
    Invoke-OwnerTests 'agentgym-clients' @($agent) @('experiments/rl/test_agentgym_native_clients_local.py')
    Invoke-OwnerTests 'webshop-boundary' @($verl) @('experiments/rl/test_webshop_native_boundary_local.py')
    Invoke-OwnerTests 'model-recipes' @($sky, "$sky/skyrl-gym") @('experiments/rl/test_model_recipe_entry_local.py', 'experiments/rl/test_official_workload_local.py')
    Invoke-OwnerTests 'qwen35-entry' @() @('experiments/rl/test_qwen35_environment_entry_local.py') $true
    Invoke-OwnerTests 'loop-environment' @("$sources/ml-loop") @('experiments/rl/test_loop_environment_local.py', 'experiments/rl/test_loop_model_entry_local.py') $true
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
