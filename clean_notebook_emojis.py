#!/usr/bin/env python3
"""
Notebook Emoji Cleanup Script
Removes all emojis except ✅ and ❌ from the notebook
"""

import json
import re
import sys

def clean_emojis(text):
    """Remove all emojis except ✅ and ❌"""
    # List of emojis to remove (keeping ✅ and ❌)
    emojis_to_remove = [
        '📊', '📁', '🚀', '🎯', '🔥', '⚠️', '🏆', '📈', '📋', '🔍', '📏', 
        '🎉', '📝', '🤖', '⚡', '🌟', '🔧', '📦', '🎨', '📖', '💡', '🔬',
        '🛠️', '⭐', '🌈', '🚨', '📰', '🔔', '💻', '📚', '🎭', '🎪', '🎨',
        '🎲', '🎳', '🏓', '🏀', '⚽', '🏈', '🎾', '🏐', '🏉', '🎱', '🏏',
        '🏑', '🏒', '🏓', '🏸', '🥊', '🥋', '🎿', '⛷️', '🏂', '🏋️', '🤺',
        '🤸', '🤽', '🚣', '🚴', '🚵', '🧘', '🏃', '🚶', '🤲', '👏', '🙌',
        '👍', '👎', '👊', '✊', '🤛', '🤜', '🤞', '✌️', '🤟', '🤘', '👌',
        '👈', '👉', '👆', '👇', '☝️', '👋', '🤚', '🖐️', '✋', '🖖', '👁️',
        '👀', '🗣️', '👤', '👥', '🫂', '👪', '👨', '👩', '👧', '👦', '👶',
        '🧠', '🫀', '🫁', '🩸', '🦠', '🧬', '🦷', '🦴', '💀', '☠️', '👻',
        '👽', '👾', '🤖', '🎃', '😄', '😃', '😀', '😊', '☺️', '😉', '😍',
        '🥰', '😘', '😗', '😙', '😚', '😋', '😛', '😝', '😜', '🤪', '🤨',
        '🧐', '🤓', '😎', '🤩', '🥳', '😏', '😒', '😞', '😔', '😟', '😕',
        '🙁', '☹️', '😣', '😖', '😫', '😩', '🥺', '😢', '😭', '😤', '😠',
        '😡', '🤬', '🤯', '😳', '🥵', '🥶', '😱', '😨', '😰', '😥', '😓',
        '🤗', '🤔', '🤭', '🤫', '🤥', '😶', '😐', '😑', '😬', '🙄', '😯',
        '😦', '😧', '😮', '😲', '🥱', '😴', '🤤', '😪', '😵', '🤐', '🥴',
        '🤢', '🤮', '🤧', '😷', '🤒', '🤕', '🤑', '🤠', '😈', '👿', '👹',
        '👺', '🤡', '💩', '👻', '💀', '☠️', '👽', '👾', '🤖', '🎃', '😺',
        '😸', '😹', '😻', '😼', '😽', '🙀', '😿', '😾'
    ]
    
    # Remove each emoji
    for emoji in emojis_to_remove:
        text = text.replace(emoji, '')
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_notebook(notebook_path):
    """Clean emojis from a Jupyter notebook"""
    print(f"Cleaning notebook: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cleaned_cells = 0
    
    # Process each cell
    for cell in notebook['cells']:
        if 'source' in cell:
            original_source = ''.join(cell['source'])
            cleaned_source = clean_emojis(original_source)
            
            if original_source != cleaned_source:
                # Split back into lines
                cell['source'] = cleaned_source.split('\n')
                # Add newlines back (except for last line)
                for i in range(len(cell['source']) - 1):
                    cell['source'][i] += '\n'
                cleaned_cells += 1
    
    # Write back the cleaned notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Cleaned {cleaned_cells} cells")
    return cleaned_cells

if __name__ == "__main__":
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "notebooks/Plagirism-Detection.ipynb"
    clean_notebook(notebook_path)