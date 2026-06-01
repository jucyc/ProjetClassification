#!/usr/bin/env python3
"""Script principal pour exécuter tous les tests avec affichage clair"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import time
import json
from datetime import datetime

def print_header(title):
    """Affiche un en-tête coloré"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_success(message):
    print(f"{message}")

def print_error(message):
    print(f"{message}")

def print_info(message):
    print(f"ℹ{message}")

def run_test(script_name, description):
    """Exécute un test et affiche les résultats"""
    print_header(description)
    
    start_time = time.time()
    
    # Exécuter le script
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    elapsed = time.time() - start_time
    
    # Afficher la sortie
    if result.stdout:
        print(result.stdout)
    
    # Afficher les erreurs éventuelles
    if result.stderr:
        print("SORTIE D'ERREUR:")
        print(result.stderr)
    
    # Déterminer le succès
    success = result.returncode == 0
    if success:
        print_success(f"Test terminé avec succès (temps: {elapsed:.2f}s)")
    else:
        print_error(f"Test échoué (code: {result.returncode}, temps: {elapsed:.2f}s)")
    
    return success, result.stdout

def main():
    print_header("SUITE DE TESTS - PROJET CLASSIFICATION D'IMAGES")
    print_info(f"Début des tests: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Vérifier que les données existent
    if not os.path.exists('tests/data/test_data.npz'):
        print_info("Génération des données de test...")
        subprocess.run([sys.executable, 'tests/generate_test_data.py'])
    
    tests = [
        ("test_linear.py", "Test du modèle linéaire"),
        ("test_mlp.py", "Test du MLP"),
    ]
    
    results = []
    
    for script, description in tests:
        script_path = os.path.join('tests', script)
        if os.path.exists(script_path):
            success, output = run_test(script_path, description)
            results.append({
                'name': description,
                'script': script,
                'success': success,
                'time': datetime.now().isoformat()
            })
        else:
            print_error(f"Script non trouvé: {script_path}")
            results.append({
                'name': description,
                'script': script,
                'success': False,
                'error': 'File not found'
            })
    
    # Résumé final
    print_header("RÉSUMÉ DES TESTS")
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for r in results:
        status = "ok" if r['success'] else "ko"
        print(f"  {status} {r['name']}")
    
    print("\n" + "-"*70)
    print(f"  Total: {passed}/{total} tests réussis")
    
    if passed == total:
        print_success("TOUS LES TESTS SONT RÉUSSIS !")
    else:
        print_error(f"{total - passed} test(s) ont échoué")
    
    print(f"\nℹFin des tests: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sauvegarder les résultats
    with open('tests/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print_info("Résultats sauvegardés dans tests/results.json")

if __name__ == "__main__":
    main()