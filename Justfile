set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

TargetDir := "E:\\Temp"
export AESDK_ROOT := justfile_directory() / "../after-effects/sdk/AfterEffectsSDK"

[windows]
build:
    cargo build --release
    Start-Process PowerShell -Verb runAs -ArgumentList "-command Copy-Item -Force '{{TargetDir}}\release\gyroflow_adobe.dll' 'C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\gyroflow_adobe.aex'"

[macos]
build:
    cargo build
    # todo
