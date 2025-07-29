"""
Script para instalar dependências necessárias
"""
import subprocess
import sys

def install_pygame():
    print("Instalando pygame...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        print("Pygame instalado com sucesso!")
    except Exception as e:
        print(f"Erro ao instalar pygame: {e}")
        print("Por favor instale manualmente usando: pip install pygame")

if __name__ == "__main__":
    install_pygame()
