# smart_city_2025

세종에서 진행되는 스마트시티 수업에 대한 내용

## notion 페이지

- wsl : [링크](https://www.notion.so/freshmea/WSL-windows-subsystem-for-linux-232123060ee780e79964ec56e36b5c18?source=copy_link)

## WSL 설치

```shellshell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

wsl --install
wsl --install -d Ubuntu-22.04
wsl --list --verbose
wsl --set-default-version 2
```

## usbipd

```shell

winget install --interactive --exact dorssel.usbipd-win
usbipd list
usbipd bind --busid 4-2
usbipd attach --wsl --busid 7-3

```

