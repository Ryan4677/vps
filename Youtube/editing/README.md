# Editing Wrapper

Executor template dinamis untuk workflow editing video.

[🇬🇧 Read in English](#english-version)

## Gambaran Umum

Memisahkan orkestrasi dari eksekusi dalam video editing workflow:
- **main.py** — mengelola paths, metadata, dan alur kontrol
- **template.py** — berisi logika editing spesifik
- **editing wrapper** — menjembatani kedua layer tersebut

## Arsitektur

```
EditingWrapper
  ├─ TemplateLoader          # Memuat file .py sebagai module
  └─ execute()               # Menyiapkan paths dan menjalankan template.apply()
```

**Class Name:** `EditingWrapper` (bukan `Editing`)  
**Import:** `from editing import EditingWrapper`

## Struktur File

```
editing/
  ├─ __init__.py            # Export EditingWrapper
  └─ editing.py             # Core implementation
```

## Struktur Direktori Output

```
output/
  ├─ clips/                 # Hasil potongan video
  │   └─ _concat_list.txt   # File temporary untuk concat
  └─ logs/                  # Execution logs
      └─ result.json        # Hasil eksekusi template
```

## Cara Pakai

### Basic Usage

```python
from editing import EditingWrapper

wrapper = EditingWrapper()
result = wrapper.execute(
    video_src="input/video.mp4",
    template_path="templates/my_template.py",
    output_dir="output",
    metadata={"title": "Video", "segments": [...]}
)
```

### Parameter `execute()`

| Parameter | Type | Required | Default | Deskripsi |
|-----------|------|----------|---------|-----------|
| `video_src` | str | Yes | - | Path ke file video sumber |
| `template_path` | str | Yes | - | Path ke file template.py |
| `output_dir` | str | No | "output" | Direktori output |
| `metadata` | Dict | No | {} | Metadata untuk template |

### Return Value

Dictionary hasil dari `template.apply()`. Struktur ditentukan oleh template.

## Persyaratan Template

File template harus mengimplementasikan fungsi `apply()`:

```python
from pathlib import Path
from typing import Dict, Any

def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Menjalankan workflow editing.
    
    Args:
        paths: Dictionary berisi semua paths yang diperlukan
        metadata: Metadata dari main.py
    
    Returns:
        Dictionary hasil (akan disimpan ke logs/result.json)
    """
    # Implementasi editing logic di sini
    pass
```

**Signature harus exact:**
- Function name: `apply`
- Parameter 1: `paths` (Dict[str, Path])
- Parameter 2: `metadata` (Dict[str, Any])
- Return: Dict[str, Any]

## Paths yang Tersedia

Dictionary `paths` yang diterima template berisi:

| Key | Type | Deskripsi |
|-----|------|-----------|
| `video_src` | Path | File video sumber (absolute path) |
| `output_dir` | Path | Direktori output utama |
| `clips_dir` | Path | Direktori untuk menyimpan klip video |
| `logs_dir` | Path | Direktori untuk menyimpan logs |
| `concat_temp` | Path | File temporary untuk concat list |

Semua paths sudah di-resolve ke absolute path.

## Alur Kerja Internal

1. **Validasi Input**
   - Cek video source exists
   - Cek template file exists
   - Cek template berekstensi .py

2. **Setup Direktori**
   - Buat `output_dir` jika belum ada
   - Buat subdirektori `clips/` dan `logs/`

3. **Load Template**
   - Import template sebagai Python module
   - Validasi fungsi `apply()` exists

4. **Prepare Paths**
   - Convert semua paths ke Path objects
   - Resolve ke absolute paths

5. **Execute Template**
   - Panggil `template.apply(paths, metadata)`
   - Tangkap return value

6. **Save Results**
   - Simpan hasil ke `logs/result.json`
   - Pretty print dengan indent

## Error Handling

### FileNotFoundError

```python
# Video tidak ditemukan
raise FileNotFoundError(f"Video tidak ditemukan: {video_src}")

# Template tidak ditemukan
raise FileNotFoundError(f"Template tidak ditemukan: {template_path}")
```

### ValueError

```python
# Template bukan file .py
raise ValueError(f"Template harus berekstensi .py, bukan {template_path.suffix}")
```

### RuntimeError

```python
# Template tidak memiliki fungsi apply()
raise RuntimeError(
    f"Template {template_path.name} harus memiliki fungsi apply()\n"
    f"Signature: def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]\n"
)
```

## Konfigurasi

Konstanta di `editing/editing.py`:

```python
LOG_DIR_NAME = "logs"        # Nama direktori logs
CLIPS_DIR_NAME = "clips"     # Nama direktori clips
```

Ubah konstanta ini jika ingin menggunakan nama direktori berbeda.

## Contoh Template

### Template Sederhana

```python
from pathlib import Path
from typing import Dict, Any

def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    video_src = paths["video_src"]
    clips_dir = paths["clips_dir"]
    
    print(f"Processing: {video_src}")
    print(f"Output to: {clips_dir}")
    
    # Logika editing Anda di sini
    # Contoh: potong video, gabung clips, dll
    
    return {
        "status": "success",
        "clips_created": 5,
        "output_path": str(clips_dir)
    }
```

### Template dengan Metadata

```python
def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    video_src = paths["video_src"]
    clips_dir = paths["clips_dir"]
    
    # Akses metadata dari main.py
    title = metadata.get("title", "Untitled")
    segments = metadata.get("segments", [])
    
    results = []
    for segment in segments:
        # Process each segment
        clip_path = clips_dir / f"clip_{segment['id']}.mp4"
        # ... editing logic ...
        results.append(str(clip_path))
    
    return {
        "status": "success",
        "title": title,
        "clips": results,
        "total_clips": len(results)
    }
```

## Output

### Console Output

```
[WRAPPER] Loading template: my_template.py
[WRAPPER] Template loaded successfully

[WRAPPER] Direktori setup:
  Video: /absolute/path/to/video.mp4
  Output: /absolute/path/to/output
  Clips: /absolute/path/to/output/clips

[WRAPPER] Executing template workflow...

[... template output ...]

[WRAPPER] Template execution complete!
Log saved: /absolute/path/to/output/logs/result.json
```

### File Output

**logs/result.json**
```json
{
  "status": "success",
  "clips_created": 5,
  "output_path": "/path/to/output/clips"
}
```

## Best Practices

### 1. Template Design

- Satu template = satu workflow spesifik
- Jangan hardcode paths, gunakan parameter `paths`
- Return informasi yang berguna untuk debugging

### 2. Error Handling

```python
def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Editing logic
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### 3. Logging

```python
def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[TEMPLATE] Starting workflow...")
    # ... logic ...
    print(f"[TEMPLATE] Completed!")
    return result
```

## Advanced Usage

### Multiple Templates

```python
wrapper = EditingWrapper()

# Template 1: Extract clips
result1 = wrapper.execute(
    video_src="input.mp4",
    template_path="templates/extract_clips.py",
    output_dir="output/step1"
)

# Template 2: Apply effects
result2 = wrapper.execute(
    video_src="output/step1/clips/final.mp4",
    template_path="templates/apply_effects.py",
    output_dir="output/step2",
    metadata=result1
)
```

### Custom Metadata

```python
metadata = {
    "title": "My Video",
    "segments": [
        {"start": 0, "stop": 1000, "label": "intro"},
        {"start": 1000, "stop": 5000, "label": "main"}
    ],
    "resolution": "1920x1080",
    "fps": 30
}

result = wrapper.execute(
    video_src="input.mp4",
    template_path="templates/smart_cut.py",
    metadata=metadata
)
```

## Troubleshooting

### Template tidak ter-load

**Problem:** `RuntimeError: Cannot load template`

**Solution:**
- Pastikan file template valid Python syntax
- Cek tidak ada circular imports
- Pastikan dependencies template ter-install

### Fungsi apply() tidak ditemukan

**Problem:** `RuntimeError: Template harus memiliki fungsi apply()`

**Solution:**
- Pastikan fungsi bernama `apply` (lowercase)
- Pastikan signature exact: `def apply(paths, metadata)`
- Jangan gunakan `@staticmethod` atau `@classmethod`

### Path tidak ditemukan

**Problem:** `FileNotFoundError`

**Solution:**
- Gunakan absolute path atau relative dari working directory
- Cek file permissions
- Pastikan file belum dipindah/dihapus

## Notes

- Template di-load sekali per eksekusi (tidak di-cache)
- Semua paths di-resolve ke absolute paths
- Struktur result dictionary bebas, ditentukan template
- Logs otomatis disimpan setelah eksekusi
- Direktori dibuat otomatis jika belum ada

---

# English Version

Dynamic template executor for video editing workflows.

[🇮🇩 Baca dalam Bahasa Indonesia](#editing-wrapper)

## Overview

Separates orchestration from execution in video editing workflows:
- **main.py** — manages paths, metadata, and control flow
- **template.py** — contains specific editing logic
- **editing wrapper** — bridges both layers

## Architecture

```
EditingWrapper
  ├─ TemplateLoader          # Loads .py files as modules
  └─ execute()               # Prepares paths and runs template.apply()
```

**Class Name:** `EditingWrapper` (not `Editing`)  
**Import:** `from editing import EditingWrapper`

## File Structure

```
editing/
  ├─ __init__.py            # Exports EditingWrapper
  └─ editing.py             # Core implementation
```

## Output Directory Structure

```
output/
  ├─ clips/                 # Video clips output
  │   └─ _concat_list.txt   # Temporary concat file
  └─ logs/                  # Execution logs
      └─ result.json        # Template execution result
```

## Usage

### Basic Usage

```python
from editing import EditingWrapper

wrapper = EditingWrapper()
result = wrapper.execute(
    video_src="input/video.mp4",
    template_path="templates/my_template.py",
    output_dir="output",
    metadata={"title": "Video", "segments": [...]}
)
```

### `execute()` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video_src` | str | Yes | - | Path to source video file |
| `template_path` | str | Yes | - | Path to template.py file |
| `output_dir` | str | No | "output" | Output directory |
| `metadata` | Dict | No | {} | Metadata for template |

### Return Value

Dictionary result from `template.apply()`. Structure is template-defined.

## Template Requirements

Template file must implement `apply()` function:

```python
from pathlib import Path
from typing import Dict, Any

def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute editing workflow.
    
    Args:
        paths: Dictionary containing all necessary paths
        metadata: Metadata from main.py
    
    Returns:
        Result dictionary (saved to logs/result.json)
    """
    # Implement editing logic here
    pass
```

**Signature must be exact:**
- Function name: `apply`
- Parameter 1: `paths` (Dict[str, Path])
- Parameter 2: `metadata` (Dict[str, Any])
- Return: Dict[str, Any]

## Available Paths

The `paths` dictionary received by template contains:

| Key | Type | Description |
|-----|------|-------------|
| `video_src` | Path | Source video file (absolute path) |
| `output_dir` | Path | Main output directory |
| `clips_dir` | Path | Directory for video clips |
| `logs_dir` | Path | Directory for logs |
| `concat_temp` | Path | Temporary concat list file |

All paths are resolved to absolute paths.

## Internal Workflow

1. **Input Validation**
   - Check video source exists
   - Check template file exists
   - Check template has .py extension

2. **Directory Setup**
   - Create `output_dir` if not exists
   - Create `clips/` and `logs/` subdirectories

3. **Load Template**
   - Import template as Python module
   - Validate `apply()` function exists

4. **Prepare Paths**
   - Convert all paths to Path objects
   - Resolve to absolute paths

5. **Execute Template**
   - Call `template.apply(paths, metadata)`
   - Capture return value

6. **Save Results**
   - Save result to `logs/result.json`
   - Pretty print with indentation

## Error Handling

### FileNotFoundError

```python
# Video not found
raise FileNotFoundError(f"Video tidak ditemukan: {video_src}")

# Template not found
raise FileNotFoundError(f"Template tidak ditemukan: {template_path}")
```

### ValueError

```python
# Template not .py file
raise ValueError(f"Template harus berekstensi .py, bukan {template_path.suffix}")
```

### RuntimeError

```python
# Template missing apply() function
raise RuntimeError(
    f"Template {template_path.name} harus memiliki fungsi apply()\n"
    f"Signature: def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]\n"
)
```

## Configuration

Constants in `editing/editing.py`:

```python
LOG_DIR_NAME = "logs"        # Logs directory name
CLIPS_DIR_NAME = "clips"     # Clips directory name
```

Modify these constants to use different directory names.

## Example Templates

### Simple Template

```python
from pathlib import Path
from typing import Dict, Any

def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    video_src = paths["video_src"]
    clips_dir = paths["clips_dir"]
    
    print(f"Processing: {video_src}")
    print(f"Output to: {clips_dir}")
    
    # Your editing logic here
    # Example: cut video, merge clips, etc.
    
    return {
        "status": "success",
        "clips_created": 5,
        "output_path": str(clips_dir)
    }
```

### Template with Metadata

```python
def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    video_src = paths["video_src"]
    clips_dir = paths["clips_dir"]
    
    # Access metadata from main.py
    title = metadata.get("title", "Untitled")
    segments = metadata.get("segments", [])
    
    results = []
    for segment in segments:
        # Process each segment
        clip_path = clips_dir / f"clip_{segment['id']}.mp4"
        # ... editing logic ...
        results.append(str(clip_path))
    
    return {
        "status": "success",
        "title": title,
        "clips": results,
        "total_clips": len(results)
    }
```

## Output

### Console Output

```
[WRAPPER] Loading template: my_template.py
[WRAPPER] Template loaded successfully

[WRAPPER] Direktori setup:
  Video: /absolute/path/to/video.mp4
  Output: /absolute/path/to/output
  Clips: /absolute/path/to/output/clips

[WRAPPER] Executing template workflow...

[... template output ...]

[WRAPPER] Template execution complete!
Log saved: /absolute/path/to/output/logs/result.json
```

### File Output

**logs/result.json**
```json
{
  "status": "success",
  "clips_created": 5,
  "output_path": "/path/to/output/clips"
}
```

## Best Practices

### 1. Template Design

- One template = one specific workflow
- Don't hardcode paths, use `paths` parameter
- Return useful information for debugging

### 2. Error Handling

```python
def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Editing logic
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### 3. Logging

```python
def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[TEMPLATE] Starting workflow...")
    # ... logic ...
    print(f"[TEMPLATE] Completed!")
    return result
```

## Advanced Usage

### Multiple Templates

```python
wrapper = EditingWrapper()

# Template 1: Extract clips
result1 = wrapper.execute(
    video_src="input.mp4",
    template_path="templates/extract_clips.py",
    output_dir="output/step1"
)

# Template 2: Apply effects
result2 = wrapper.execute(
    video_src="output/step1/clips/final.mp4",
    template_path="templates/apply_effects.py",
    output_dir="output/step2",
    metadata=result1
)
```

### Custom Metadata

```python
metadata = {
    "title": "My Video",
    "segments": [
        {"start": 0, "stop": 1000, "label": "intro"},
        {"start": 1000, "stop": 5000, "label": "main"}
    ],
    "resolution": "1920x1080",
    "fps": 30
}

result = wrapper.execute(
    video_src="input.mp4",
    template_path="templates/smart_cut.py",
    metadata=metadata
)
```

## Troubleshooting

### Template cannot be loaded

**Problem:** `RuntimeError: Cannot load template`

**Solution:**
- Ensure template file has valid Python syntax
- Check for circular imports
- Ensure template dependencies are installed

### apply() function not found

**Problem:** `RuntimeError: Template harus memiliki fungsi apply()`

**Solution:**
- Ensure function is named `apply` (lowercase)
- Ensure exact signature: `def apply(paths, metadata)`
- Don't use `@staticmethod` or `@classmethod`

### Path not found

**Problem:** `FileNotFoundError`

**Solution:**
- Use absolute path or relative from working directory
- Check file permissions
- Ensure file hasn't been moved/deleted

## Notes

- Template is loaded once per execution (not cached)
- All paths are resolved to absolute paths
- Result dictionary structure is template-defined
- Logs are automatically saved after execution
- Directories are created automatically if not exist