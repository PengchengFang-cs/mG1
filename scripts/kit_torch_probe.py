import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args = parser.parse_args()
app = AppLauncher(args).app
import sys, torch
print("KIT torch:", torch.__version__, torch.__file__)
print("KIT sys.path torch-ish:", [p for p in sys.path if "torch" in p.lower()][:10])
try:
    import torchvision; print("KIT torchvision:", torchvision.__version__, torchvision.__file__)
except Exception as e:
    print("KIT torchvision import failed:", repr(e)[:200])
app.close()
