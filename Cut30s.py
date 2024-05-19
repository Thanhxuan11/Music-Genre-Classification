import librosa
import os
from scipy.io.wavfile import write

def cut_audio_from_folder(input_folder, output_folder, segment_duration=30):
    """
    Cắt 30s đầu của các bài hát trong thư mục đầu vào và lưu vào thư mục đầu ra với định dạng .wav.

    Args:
        input_folder: Đường dẫn đến thư mục đầu vào chứa các bài hát.
        output_folder: Đường dẫn đến thư mục đầu ra để lưu trữ các đoạn âm thanh cắt ra.
        segment_duration: Thời lượng cắt (tính bằng giây).
    """

    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Lặp qua tất cả các tệp trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav') or filename.lower().endswith('.mp3') or filename.lower().endswith('.flac'):
            # Đường dẫn đầy đủ đến tệp nhạc đầu vào
            input_path = os.path.join(input_folder, filename)

            # Tạo tên tệp đầu ra với định dạng `ten_file_30s.wav`
            output_filename = f"{filename[:-4]}_30s.wav"  # Save as WAV
            output_path = os.path.join(output_folder, output_filename)

            # Đọc và cắt 30s đầu của bài hát
            y, sr = librosa.load(input_path, sr=None)

            # Xác định số lượng mẫu cần cắt (30 giây)
            num_samples = int(segment_duration * sr)

            # Cắt 30s đầu tiên và lưu vào file riêng
            segment = y[:num_samples]
            write(output_path, sr, segment)

            print(f"Đã cắt 30s đầu của bài hát {filename} và lưu vào {output_folder}")

if __name__ == "__main__":
    # Thay đổi đường dẫn thư mục đầu vào và thư mục đầu ra
    input_folder = "C:/Users/xuanl/Downloads"
    output_folder = "Music/Cach_Mang"

    # Gọi hàm để cắt 30s đầu của các bài hát và lưu thành .wav
    cut_audio_from_folder(input_folder, output_folder)
