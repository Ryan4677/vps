import json
import pysubs2
import subprocess
import os
import re
import sys

def milliseconds_to_seconds(ms):
    """Konversi milidetik ke detik"""
    return ms / 1000

def get_transcript_for_segment(data, segment_id):
    """
    Ambil transcript yang sesuai dengan segment berdasarkan start dan stop
    """
    # Cari segment berdasarkan segment_id
    segment = None
    for candidate in data['candidates']:
        if candidate['segment_id'] == segment_id:
            segment = candidate
            break
    
    if not segment:
        print(f"Segment {segment_id} tidak ditemukan!")
        return [], 0, 0
    
    segment_start = segment['start']
    segment_stop = segment['stop']
    
    print(f"Segment {segment_id}: {segment['title']}")
    print(f"Start: {segment_start}ms ({milliseconds_to_seconds(segment_start)}s)")
    print(f"Stop: {segment_stop}ms ({milliseconds_to_seconds(segment_stop)}s)")
    print(f"Duration: {segment['duration']}ms\n")
    
    # Filter transcript yang berada dalam range segment
    filtered_transcript = []
    for trans in data['transcript']:
        # Cek apakah transcript berada dalam range segment
        if trans['start'] >= segment_start and trans['stop'] <= segment_stop:
            filtered_transcript.append(trans)
    
    print(f"Ditemukan {len(filtered_transcript)} subtitle dalam segment ini\n")
    return filtered_transcript, segment_start, segment_stop

def create_ass_subtitle(transcript_data, segment_start, output_ass_path):
    """
    Buat file subtitle ASS menggunakan pysubs2 dengan style CapCut modern
    """
    # Buat SSAFile baru
    subs = pysubs2.SSAFile()
    
    # Buat style custom untuk subtitle - Style Podcast Professional
    style = pysubs2.SSAStyle()
    style.fontname = "Arial"  # Sans-serif standar untuk keterbacaan maksimal
    style.fontsize = 18  # Size optimal untuk podcast (berdasarkan Section 508 standards)
    style.bold = False  # Tidak bold untuk tampilan lebih clean
    style.primary_color = pysubs2.Color(255, 255, 255)  # Putih bersih
    style.back_color = pysubs2.Color(0, 0, 0, 180)  # Hitam semi-transparent untuk kontras
    style.outline_color = pysubs2.Color(0, 0, 0)  # Border hitam
    style.outline = 2  # Border sedang untuk keterbacaan
    style.shadow = 0  # Tanpa shadow untuk tampilan lebih minimalis
    style.alignment = 2  # Bottom center (standard positioning)
    style.margin_v = 20  # Margin kecil dari bawah (tetap di bagian bawah)
    style.margin_l = 60  # Margin kiri lebih besar (max 45 chars per line)
    style.margin_r = 60  # Margin kanan lebih besar
    
    subs.styles["Default"] = style
    
    # Tambahkan setiap transcript sebagai event
    for trans in transcript_data:
        # Hitung waktu relatif terhadap segment start
        relative_start = trans['start'] - segment_start
        relative_stop = trans['stop'] - segment_start
        
        # Format text: uppercase untuk emphasis (opsional)
        text = trans['text'].strip()
        
        # Buat event (subtitle entry)
        event = pysubs2.SSAEvent(
            start=relative_start,
            end=relative_stop,
            text=text,
            style="Default"
        )
        subs.events.append(event)
    
    # Simpan sebagai file ASS
    subs.save(output_ass_path)
    print(f"✓ File subtitle berhasil dibuat: {output_ass_path}")

def parse_ffmpeg_progress(line, duration_seconds):
    """
    Parse output FFmpeg untuk mendapatkan progress
    """
    # Cari pattern time=HH:MM:SS.MS
    time_match = re.search(r'time=(\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
    if time_match:
        hours = int(time_match.group(1))
        minutes = int(time_match.group(2))
        seconds = float(time_match.group(3))
        
        current_time = hours * 3600 + minutes * 60 + seconds
        
        if duration_seconds > 0:
            progress = min((current_time / duration_seconds) * 100, 100)
            return progress
    return None

def print_progress_bar(progress, prefix='Progress', length=40):
    """
    Tampilkan progress bar di console
    """
    filled = int(length * progress / 100)
    bar = '█' * filled + '░' * (length - filled)
    sys.stdout.write(f'\r{prefix}: |{bar}| {progress:.1f}%')
    sys.stdout.flush()

def get_video_duration(video_path):
    """
    Dapatkan durasi video dalam detik menggunakan FFprobe
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return 0

def add_subtitles_to_video_ffmpeg(video_path, ass_path, output_path):
    """
    Burn subtitle ke video menggunakan FFmpeg dengan progress indicator
    """
    print(f"\nMemproses video dengan FFmpeg...")
    print(f"Input video: {video_path}")
    print(f"Subtitle file: {ass_path}")
    print(f"Output video: {output_path}\n")
    
    # Dapatkan durasi video
    duration = get_video_duration(video_path)
    if duration == 0:
        print("⚠ Tidak dapat mendeteksi durasi video, progress mungkin tidak akurat")
    
    # Escape path untuk Windows (ganti backslash dengan forward slash dan escape colon)
    ass_path_escaped = ass_path.replace('\\', '/').replace(':', '\\:')
    
    # Command FFmpeg untuk burn subtitle dengan kualitas tinggi
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"ass={ass_path_escaped}",
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-c:a', 'copy',
        '-progress', 'pipe:1',  # Output progress ke stdout
        '-y',
        output_path
    ]
    
    try:
        # Jalankan FFmpeg dengan real-time output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("🎬 Memulai proses encoding...")
        
        # Baca output line by line
        for line in process.stdout:
            progress = parse_ffmpeg_progress(line, duration)
            if progress is not None:
                print_progress_bar(progress, prefix='Encoding')
        
        # Tunggu proses selesai
        process.wait()
        
        # Print newline setelah progress bar
        print()
        
        if process.returncode == 0:
            print(f"\n✓ Video dengan subtitle berhasil dibuat: {output_path}")
            return True
        else:
            print(f"\n✗ Error saat menjalankan FFmpeg (return code: {process.returncode})")
            return False
            
    except FileNotFoundError:
        print("\n✗ Error: FFmpeg tidak ditemukan!")
        print("Pastikan FFmpeg sudah terinstall dan ada di PATH")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def cut_video_with_progress(video_path, start_sec, stop_sec, output_path):
    """
    Potong video dengan progress indicator
    """
    print(f"\n✂ Memotong video dari {start_sec}s sampai {stop_sec}s...")
    
    duration = stop_sec - start_sec
    
    command = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(start_sec),
        '-to', str(stop_sec),
        '-c', 'copy',
        '-progress', 'pipe:1',
        '-y',
        output_path
    ]
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("🎬 Memotong video...")
        
        for line in process.stdout:
            progress = parse_ffmpeg_progress(line, duration)
            if progress is not None:
                print_progress_bar(progress, prefix='Cutting')
        
        process.wait()
        print()  # Newline after progress bar
        
        if process.returncode == 0:
            print(f"✓ Video berhasil dipotong: {output_path}")
            return True
        else:
            print(f"✗ Error saat memotong video (return code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"✗ Error saat memotong video: {e}")
        return False

def load_json_from_file(json_path):
    """
    Load data JSON dari file
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Berhasil membaca file: {json_path}\n")
        return data
    except FileNotFoundError:
        print(f"✗ Error: File {json_path} tidak ditemukan!")
        return None
    except json.JSONDecodeError as e:
        print(f"✗ Error: File JSON tidak valid! {e}")
        return None

def add_subtitles_to_video(video_path, json_data, segment_id, output_path, use_full_video=False):
    """
    Tambahkan subtitle ke video berdasarkan segment yang dipilih
    
    Parameters:
    - use_full_video: True jika video adalah video lengkap yang perlu dipotong
                      False jika video sudah berupa clip segment
    """
    # Parse JSON jika masih string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Ambil transcript untuk segment
    transcript_data, segment_start, segment_stop = get_transcript_for_segment(data, segment_id)
    
    if not transcript_data:
        print("Tidak ada subtitle untuk ditambahkan!")
        return
    
    # Tentukan video yang akan diproses
    video_to_process = video_path
    segment_start_for_subtitle = segment_start
    
    # Jika use_full_video, potong video terlebih dahulu
    if use_full_video:
        segment_start_sec = milliseconds_to_seconds(segment_start)
        segment_stop_sec = milliseconds_to_seconds(segment_stop)
        
        # Buat temporary video yang sudah dipotong
        temp_video = "temp_cut_video.mp4"
        
        success = cut_video_with_progress(video_path, segment_start_sec, segment_stop_sec, temp_video)
        
        if success:
            video_to_process = temp_video
            segment_start_for_subtitle = 0  # Reset karena video dimulai dari 0
        else:
            return
    else:
        print("\n📹 Video sudah berupa clip segment, tidak perlu dipotong...")
        segment_start_for_subtitle = 0
    
    # Buat file subtitle ASS
    ass_file = "temp_subtitle.ass"
    print("\n📝 Membuat file subtitle...")
    create_ass_subtitle(transcript_data, segment_start, ass_file)
    
    # Burn subtitle ke video menggunakan FFmpeg
    success = add_subtitles_to_video_ffmpeg(video_to_process, ass_file, output_path)
    
    # Cleanup temporary files
    print("\n🧹 Membersihkan file temporary...")
    if use_full_video and os.path.exists("temp_cut_video.mp4"):
        os.remove("temp_cut_video.mp4")
        print("✓ File temporary video dihapus")
    
    if os.path.exists(ass_file):
        os.remove(ass_file)
        print("✓ File temporary subtitle dihapus")
    
    if success:
        print(f"\n{'='*50}")
        print("✨ SELESAI! ✨")
        print(f"{'='*50}")
        print(f"📁 Output: {output_path}")

# Contoh penggunaan
if __name__ == "__main__":
    # Path file
    json_file = "output/4c24febd-247f-400a-aba6-4b1fc36fd488/metadata/metadata.json"
    input_video = "output/4c24febd-247f-400a-aba6-4b1fc36fd488/video/output/clips/clip_001_final.mp4"
    output_video = "output_with_subtitles.mp4"
    
    # Pilih segment (1-5)
    SEGMENT = 1
    
    # Load data JSON dari file
    print("=" * 50)
    print("🎬 VIDEO SUBTITLE GENERATOR")
    print("=" * 50)
    json_data = load_json_from_file(json_file)
    
    if json_data:
        # Jalankan
        add_subtitles_to_video(input_video, json_data, SEGMENT, output_video, use_full_video=False)
    else:
        print("Program dihentikan karena error membaca file JSON.")