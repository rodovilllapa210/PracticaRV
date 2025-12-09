"""
Script para ejecutar PracticaRV.py con TODOS los ISINs disponibles
"""

import sys
import os

# Modify PracticaRV.py to use all ISINs
print("=" * 80)
print("CONFIGURANDO PARA PROCESAR TODOS LOS ISINs")
print("=" * 80)

# Read the file
with open('PracticaRV.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace USE_ALL_ISINS = False with True
if 'USE_ALL_ISINS = False' in content:
    content = content.replace('USE_ALL_ISINS = False', 'USE_ALL_ISINS = True')
    
    with open('PracticaRV.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Configuración actualizada: USE_ALL_ISINS = True")
    print("\nEjecutando PracticaRV.py con todos los ISINs...")
    print("Esto puede tardar 8-10 minutos...")
    print("=" * 80)
    print()
    
    # Execute PracticaRV.py
    os.system('python PracticaRV.py')
    
    # Restore original setting
    content = content.replace('USE_ALL_ISINS = True', 'USE_ALL_ISINS = False')
    with open('PracticaRV.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✓ Configuración restaurada: USE_ALL_ISINS = False")
    
else:
    print("⚠ USE_ALL_ISINS ya está configurado como True")
    print("Ejecutando PracticaRV.py...")
    os.system('python PracticaRV.py')
