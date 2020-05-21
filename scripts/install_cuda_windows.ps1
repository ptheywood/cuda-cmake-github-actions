

# CUDA Version to download and install. @todo - make this be read in from the matrix.
$CUDA_VERSION_FULL = "10.2.89"

# Dictionary of known cuda versions and thier download URLS, which do not follow a consistent pattern :(
$CUDA_KNOWN_URLS = @{
    "8.0.44" = "http://developer.nvidia.com/compute/cuda/8.0/Prod/network_installers/cuda_8.0.44_win10_network-exe";
    "8.0.61" = "http://developer.nvidia.com/compute/cuda/8.0/Prod2/network_installers/cuda_8.0.61_win10_network-exe";
    "9.0.176" = "http://developer.nvidia.com/compute/cuda/9.0/Prod/network_installers/cuda_9.0.176_win10_network-exe";
    "9.1.85" = "http://developer.nvidia.com/compute/cuda/9.1/Prod/network_installers/cuda_9.1.85_win10_network";
    "9.2.148" = "http://developer.nvidia.com/compute/cuda/9.2/Prod2/network_installers2/cuda_9.2.148_win10_network";
    "10.0.130" = "http://developer.nvidia.com/compute/cuda/10.0/Prod/network_installers/cuda_10.0.130_win10_network";
    "10.1.105" = "http://developer.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.105_win10_network.exe";
    "10.1.168" = "http://developer.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.168_win10_network.exe";
    "10.1.243" = "http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe";
    "10.2.89" = "http://developer.download.nvidia.com/compute/cuda/10.2/Prod/network_installers/cuda_10.2.89_win10_network.exe";
}


## -----------------
## Prepare Variables
## -----------------

# Validate CUDA version, extracting components via regex
$cuda_ver_matched = $CUDA_VERSION_FULL -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$"
if(-not $cuda_ver_matched){
    Write-Host "Invalid CUDA version specified, <major>.<minor>.<patch> required. '$CUDA_VERSION_FULL'."
    exit 1
}
$CUDA_MAJOR=$Matches.major
$CUDA_MINOR=$Matches.minor
$CUDA_PATCH=$Matches.patch


# Build CUDA related variables.
#  If the specified version is in the known addresses, use that one. 
$CUDA_REPO_PKG_REMOTE=""
if($CUDA_KNOWN_URLS.containsKey($CUDA_VERSION_FULL)){
    $CUDA_REPO_PKG_REMOTE=$CUDA_KNOWN_URLS[$CUDA_VERSION_FULL]
} else{
    # Guess what the url is given the most recent pattern (at the time of writing, 10.1)
    Write-Host "note: URL for CUDA ${$CUDA_VERSION_FULL} not known, estimating."
    $CUDA_REPO_PKG_REMOTE="http://developer.download.nvidia.com/compute/cuda/$($CUDA_MAJOR).$($CUDA_MINOR)/Prod/network_installers/cuda_$($CUDA_VERSION_FULL)_win10_network.exe"
}
$CUDA_REPO_PKG_LOCAL="cuda_$($CUDA_VERSION_FULL)_win10_network.exe"


# Build list of required cuda packages to be installed. See https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#install-cuda-software for pacakge details. 

# @todo - make this configurable outside the script and therefore be fed in? Although ti is CUDA version dependent...

# CUDA < 9.1 had a differnt package name for the compiler.
$NVCC_PACKAGE_NAME="nvcc"
if ([version]$CUDA_VERSION_FULL -lt [version]"9.1"){
    $NVCC_PACKAGE_NAME="compiler"
}
# Build string containing list of pacakges. Do not need Display.Driver
$CUDA_PACKAGES  = "$($NVCC_PACKAGE_NAME)_$($CUDA_MAJOR).$($CUDA_MINOR)"
# $CUDA_PACKAGES += " visual_studio_integration_$($CUDA_MAJOR).$($CUDA_MINOR)"
# $CUDA_PACKAGES += " curand_dev_$($CUDA_MAJOR).$($CUDA_MINOR)"
# $CUDA_PACKAGES += " nvrtc_$($CUDA_MAJOR).$($CUDA_MINOR)"
# $CUDA_PACKAGES += " nvrtc_dev_$($CUDA_MAJOR).$($CUDA_MINOR)"
# $CUDA_PACKAGES += " cupti_$($CUDA_MAJOR).$($CUDA_MINOR)"



## ------------
## Install CUDA
## ------------

# Get CUDA network installer
Write-Host "Downloading CUDA Network Installer for $($CUDA_VERSION_FULL) from: $($CUDA_REPO_PKG_REMOTE)"
Invoke-WebRequest $CUDA_REPO_PKG_REMOTE -OutFile $CUDA_REPO_PKG_LOCAL | Out-Null
if(Test-Path -Path $CUDA_REPO_PKG_LOCAL){
    Write-Host "Downloading Complete"
} else {
    Write-Host "Error: Failed to download $($CUDA_REPO_PKG_LOCAL) from $($CUDA_REPO_PKG_REMOTE)"
    exit 1
}

# Invoke silent install of CUDA (via network installer)
Write-Host "Installing CUDA $($CUDA_VERSION_FULL) Compiler and Runtime"
Start-Process -Wait -FilePath .\"$($CUDA_REPO_PKG_LOCAL)" -ArgumentList "-s $($CUDA_PACKAGES)"

# Check the return status of the CUDA installer.
if ($? -eq $false) {
    Write-Host "Error: CUDA installer reported error. $($LASTEXITCODE)"
    exit 1 
}

# @todo - set environment variables like path.


# The silent cuda installer doesn't set the PATH with the currently specified subpackages, so must set CUDA_PATH and PATH manually. This is in part a workaround for CMAKE < 3.17 not correctly setting some values when using visual studio generators.
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$($CUDA_MAJOR).$($CUDA_MINOR)"
$PATH_CUDA_PATH = "$($CUDA_PATH)\bin\;$($CUDA_PATH)\libnvvp\;"
# Set environmental variables in this session
Write-Host "CUDA_PATH $($CUDA_PATH)"
Write-Host "Setting CUDA_PATH and PATH."
$env:CUDA_PATH = "$($CUDA_PATH)"
$env:PATH = "$($PATH_CUDA_PATH)$($env:PATH)"
# Make the new values persist the reboot via the registry.
[Environment]::SetEnvironmentVariable("CUDA_PATH", $env:CUDA_PATH, [System.EnvironmentVariableTarget]::Machine)
[Environment]::SetEnvironmentVariable("PATH", $env:PATH, [System.EnvironmentVariableTarget]::Machine)
# Note that these update the registry, and do not effect the current session until a restart.

nvcc -V

# Add to github actions.
Write-Host "::add-path::${$CUDA_PATH}\bin"
Write-Host "::add-path::${$CUDA_PATH}\libnvvp"
Write-Host "::set-env name=CUDA_PATH::${$CUDA_PATH}"

Write-Host "Installation Complete!"