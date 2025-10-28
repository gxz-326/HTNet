"""
Simple syntax checker for the enhanced model
"""

import ast
import sys

def check_file_syntax(filepath):
    """Check Python file for syntax errors"""
    print(f"Checking: {filepath}")
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✓ Syntax OK")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    files_to_check = [
        'Model.py',
        'train_enhanced_facial_palsy.py',
        'visualize_roi_and_asymmetry.py',
        'test_enhanced_model.py'
    ]
    
    print("="*60)
    print("Checking Python syntax for enhanced modules")
    print("="*60 + "\n")
    
    all_ok = True
    for f in files_to_check:
        if not check_file_syntax(f):
            all_ok = False
        print()
    
    if all_ok:
        print("="*60)
        print("✓ All files have valid Python syntax!")
        print("="*60)
    else:
        print("="*60)
        print("✗ Some files have syntax errors")
        print("="*60)
        sys.exit(1)

if __name__ == '__main__':
    main()
