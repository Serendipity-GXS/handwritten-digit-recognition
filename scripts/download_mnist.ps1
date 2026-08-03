# download_mnist.ps1
# Download the MNIST dataset (4 files, idx format) and verify sizes.
# Same files as yann.lecun.com/exdb/mnist, hosted on multiple mirrors.
#  - googleapis (default, fast in CN): https://storage.googleapis.com/cvdf-datasets/mnist
#  - s3        (fallback):             https://ossci-datasets.s3.amazonaws.com/mnist
#  - official                        : http://yann.lecun.com/exdb/mnist
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/download_mnist.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/download_mnist.ps1 -Source s3

param(
    [ValidateSet("googleapis", "s3", "official")]
    [string]$Source = "googleapis"
)

$ErrorActionPreference = "Stop"

$base    = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $base "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

switch ($Source) {
    "googleapis" { $baseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist" }
    "s3"         { $baseUrl = "https://ossci-datasets.s3.amazonaws.com/mnist" }
    "official"   { $baseUrl = "http://yann.lecun.com/exdb/mnist" }
}

$files = @(
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
)

# 1) Download (skip files that already exist)
foreach ($f in $files) {
    $dest = Join-Path $dataDir $f
    if (Test-Path $dest) { Write-Host "[skip] $f already exists"; continue }
    Write-Host "[get ] $f"
    & curl.exe -L --fail --retry 5 --retry-all-errors --retry-delay 2 -o $dest "$baseUrl/$f"
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $f" }
}

# 2) Decompress .gz -> .idx, then remove the .gz
# Note: use .NET GzipStream (works for plain-gzip, no external deps).
foreach ($f in $files) {
    $gz  = Join-Path $dataDir $f
    if (-not (Test-Path $gz)) { continue }
    $idx = [System.IO.Path]::ChangeExtension($gz, "")
    if (Test-Path $idx) { Write-Host "[skip] $idx already exists"; continue }
    Write-Host "[unzip] $f"
    try {
        $in    = [System.IO.File]::OpenRead($gz)
        $out   = [System.IO.File]::Create($idx)
        $gzip  = New-Object System.IO.Compression.GzipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
        $gzip.CopyTo($out)
        $gzip.Dispose(); $out.Dispose(); $in.Dispose()
    } catch {
        throw "Unzip failed: $f"
    }
    Remove-Item $gz
}

# 3) Verify sizes (uncompressed byte counts)
$expected = @{
    "train-images-idx3-ubyte" = 47040016
    "train-labels-idx1-ubyte" = 60008
    "t10k-images-idx3-ubyte"  = 7840016
    "t10k-labels-idx1-ubyte"  = 10008
}

Write-Host ""
Write-Host "=== verification ==="
$allOk = $true
foreach ($k in $expected.Keys) {
    $p = Join-Path $dataDir $k
    if (Test-Path $p) {
        $len = (Get-Item $p).Length
        if ($len -eq $expected[$k]) {
            Write-Host ("{0,-28} {1,12} bytes  OK" -f $k, $len)
        } else {
            Write-Host ("{0,-28} {1,12} bytes  MISMATCH (expect {2})" -f $k, $len, $expected[$k])
            $allOk = $false
        }
    } else {
        Write-Host ("{0,-28} MISSING" -f $k)
        $allOk = $false
    }
}

if ($allOk) {
    Write-Host "ALL FILES VERIFIED."
} else {
    Write-Host "SOME FILES FAILED - check network / file integrity."
    exit 1
}
