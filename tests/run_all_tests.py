import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import time
import json
from datetime import datetime

# CORRECTION : Forcer l'encodage UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_success(message):
    print(f"[OK] {message}")

def print_error(message):
    print(f"[FAIL] {message}")

def print_info(message):
    print(f"ℹ {message}")

def run_test(script_name, description):
    """Exécute un test et affiche les résultats"""
    print_header(description)
    
    start_time = time.time()
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    
    if not os.path.exists(script_path):
        print_error(f"Script non trouve: {script_path}")
        return False, ""
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    elapsed = time.time() - start_time
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print("SORTIE D'ERREUR:")
        print(result.stderr)
    
    success = result.returncode == 0
    if success:
        print_success(f"Test termine (temps: {elapsed:.2f}s)")
    else:
        print_error(f"Test echoue (code: {result.returncode})")
    
    return success, result.stdout

def main():
    print_header("SUITE DE TESTS - PROJET CLASSIFICATION D'IMAGES")
    print_info(f"De but des tests: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CORRECTION : Utiliser le bon chemin
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Vérifier que les données existent
    if not os.path.exists(os.path.join(tests_dir, "data", "test_data.npz")):
        print_info("Generation des donnees de test...")
        subprocess.run([sys.executable, "generate_test_data.py"], cwd=tests_dir)
    
    tests = [
        ("test_linear.py", "Test du modele lineaire"),
        ("test_mlp.py", "Test du MLP"),
    ]
    
    results = []
    
    for script, description in tests:
        script_path = os.path.join(tests_dir, script)
        if os.path.exists(script_path):
            success, output = run_test(script, description)
            results.append({
                'name': description,
                'script': script,
                'success': success,
                'time': datetime.now().isoformat()
            })
        else:
            print_error(f"Script non trouve: {script_path}")
            results.append({
                'name': description,
                'script': script,
                'success': False,
                'error': 'File not found'
            })
    
    print_header("RESUME DES TESTS")
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for r in results:
        status = "[OK]" if r['success'] else "[FAIL]"
        print(f"  {status} {r['name']}")
    
    print("\n" + "-"*70)
    print(f"  Total: {passed}/{total} tests reussis")
    
    if passed == total:
        print_success("TOUS LES TESTS SONT REUSSIS !")
    else:
        print_error(f"{total - passed} test(s) ont echoue")
    
    print(f"\nℹ Fin des tests: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CORRECTION : Sauvegarder dans le bon dossier
    results_file = os.path.join(tests_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print_info(f"Resultats sauvegardes dans {results_file}")

if __name__ == "__main__":
    main()