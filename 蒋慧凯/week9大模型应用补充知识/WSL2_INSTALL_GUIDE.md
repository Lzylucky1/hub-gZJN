# WSL2 + Ubuntu 22.04 安装指南

> 由于 vLLM 不支持 Windows 原生环境，必须在 WSL2（Windows Subsystem for Linux）中运行。本指南提供逐步安装步骤。
> 
> 注意：WSL2 安装需要管理员权限和一次系统重启，无法由脚本自动完成，请按以下步骤手动操作。

---

## 一、启用 WSL2

### 方式 A：自动安装（推荐，Windows 10/11 都支持）

以**管理员身份**打开 PowerShell，执行：

```powershell
wsl --install -d Ubuntu
```

执行后系统会提示重启，**重启电脑**。

重启后再次打开 PowerShell，继续执行：

```powershell
wsl --install -d Ubuntu
```

这次会完成 Ubuntu 的安装（默认最新 LTS，支持 vLLM），按提示创建用户名和密码。

### 方式 B：分步启用（如果方式 A 失败）

```powershell
# 1. 启用 WSL
wsl --install --no-distribution

# 2. 启用虚拟机平台
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 3. 设置 WSL 默认版本为 2
wsl --set-default-version 2

# 4. 从 Microsoft Store 搜索 "Ubuntu 22.04" 安装
```

完成后重启电脑。

> 注意：如果你需要严格使用 Ubuntu 22.04，可在 Microsoft Store 中搜索 "Ubuntu 22.04.3 LTS" 单独安装。本作业对 Ubuntu 版本没有严格要求，22.04 / 24.04 均可运行 vLLM。

---

## 二、验证 WSL2 安装

```powershell
wsl -l -v
```

预期输出：

```
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

确保 `VERSION` 是 `2`，不是 `1`。

---

## 三、进入 WSL2 并更新系统

```bash
wsl
# 或者在 PowerShell 中执行：wsl ~

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv build-essential git curl wget
```

---

## 四、配置 CUDA 支持（WSL2 会自动使用 Windows 驱动）

WSL2 中的 CUDA 不需要单独安装驱动，只要 Windows 侧 NVIDIA 驱动正常即可。

在 WSL2 中验证：

```bash
nvidia-smi
```

应该能看到与 Windows 侧相同的 GPU 信息。

---

## 五、后续步骤

WSL2 安装完成后，继续按 `user_guide.md` 中的步骤：

1. 创建 Python 虚拟环境
2. 安装 `requirements.txt`
3. 启动 vLLM server
4. 运行 `run_all.py`

---

## 六、常见问题

### 6.1 BIOS 中未启用虚拟化

如果安装失败，请进入 BIOS 开启：
- Intel CPU：`Intel VT-x` / `Virtualization Technology`
- AMD CPU：`SVM Mode` / `AMD-V`

### 6.2 WSL2 内核版本过低

```powershell
wsl --update
```

### 6.3 WSL2 占用 C 盘空间过大

```powershell
# 查看 WSL 发行版位置
wsl -l -v

# 导出到其他盘
wsl --export Ubuntu-22.04 D:\wsl\ubuntu.tar
wsl --unregister Ubuntu-22.04
wsl --import Ubuntu-22.04 D:\wsl\Ubuntu-22.04 D:\wsl\ubuntu.tar --version 2
```

---

## 七、参考链接

- 微软官方文档：https://learn.microsoft.com/zh-cn/windows/wsl/install
- NVIDIA WSL CUDA 指南：https://docs.nvidia.com/cuda/wsl-user-guide/
