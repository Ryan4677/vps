#editing/editing.py

import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

# ============ WRAPPER VARIABLES ============

LOG_DIR_NAME = "logs"
CLIPS_DIR_NAME = "clips"

# ============ TEMPLATE LOADER ============

class TemplateLoader:
    """Load template dari file .py"""
    
    @staticmethod
    def load(template_path: Path) -> Any:
        """
        Load template dari file .py
        
        Args:
            template_path: Path ke file template.py
        
        Returns:
            Module template yang sudah di-load
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template tidak ditemukan: {template_path}")
        
        if template_path.suffix != ".py":
            raise ValueError(f"Template harus berekstensi .py, bukan {template_path.suffix}")
        
        print(f"\n[WRAPPER] Loading template: {template_path.name}")
        
        spec = importlib.util.spec_from_file_location("editing_template", template_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load template: {template_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, "apply"):
            raise RuntimeError(
                f"Template {template_path.name} harus memiliki fungsi apply()\n"
                f"Signature: def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]\n"
            )
        
        print(f"[WRAPPER] Template loaded successfully")
        return module

# ============ EDITING WRAPPER CLASS ============

class EditingWrapper:
    """Main wrapper class untuk manage template execution."""
    
    def __init__(self):
        """Initialize wrapper."""
        self.loader = TemplateLoader()
    
    def execute(self, 
                video_src: str,
                template_path: str,
                output_dir: str = "output",
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute template dengan paths dari main.py.
        
        Args:
            video_src: Path video source
            template_path: Path ke file template.py
            output_dir: Direktori output (default: output/)
            metadata: Dict metadata dari main.py (default: {})
        
        Returns:
            Hasil dari template.apply()
        """
        # Convert to Path
        video_src = Path(video_src)
        template_path = Path(template_path)
        output_dir = Path(output_dir)
        metadata = metadata or {}
        
        # Validasi
        if not video_src.exists():
            raise FileNotFoundError(f"Video tidak ditemukan: {video_src}")
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template tidak ditemukan: {template_path}")
        
        # Create directories
        self._ensure_dirs(output_dir)
        
        # Load template
        template = self.loader.load(template_path)
        
        # Prepare paths
        paths = self._prepare_paths(video_src, output_dir)
        
        print(f"\n[WRAPPER] Direktori setup:")
        print(f"  Video: {paths['video_src']}")
        print(f"  Output: {paths['output_dir']}")
        print(f"  Clips: {paths['clips_dir']}")
        
        print(f"\n[WRAPPER] Executing template workflow...\n")
        
        # Execute template
        result = template.apply(paths, metadata)
        
        print(f"\n[WRAPPER] Template execution complete!")
        
        # Save result ke log
        self._save_log(result, output_dir)
        
        return result
    
    @staticmethod
    def _ensure_dirs(output_dir: Path):
        """Buat direktori yang diperlukan."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / CLIPS_DIR_NAME).mkdir(exist_ok=True)
        (output_dir / LOG_DIR_NAME).mkdir(exist_ok=True)
    
    @staticmethod
    def _prepare_paths(video_src: Path, output_dir: Path) -> Dict[str, Path]:
        """Return semua paths yang diperlukan."""
        return {
            "video_src": video_src,
            "output_dir": output_dir,
            "clips_dir": output_dir / CLIPS_DIR_NAME,
            "logs_dir": output_dir / LOG_DIR_NAME,
            "concat_temp": output_dir / CLIPS_DIR_NAME / "_concat_list.txt",
        }
    
    @staticmethod
    def _save_log(result: Dict[str, Any], output_dir: Path):
        """Save result ke log file."""
        log_file = output_dir / LOG_DIR_NAME / "result.json"
        log_file.write_text(json.dumps(result, indent=2))
        print(f"Log saved: {log_file}")