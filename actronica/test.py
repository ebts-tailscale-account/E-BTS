import tkinter as tk
import sounddevice as sd
import numpy as np

# --- Глобальные параметры генератора ---
SAMPLE_RATE = 48000

current_frequency = 65.0  # Стартовая частота (Hz) - идеальна для HapCoil
current_amplitude = 0.5   # Стартовая амплитуда (0.0 - 1.0)
current_phase = 0.0       # Фазовый аккумулятор для плавных переходов

def audio_callback(outdata, frames, time, status):
    """Функция, которая непрерывно генерирует аудио-буфер для HSD"""
    global current_phase, current_frequency, current_amplitude
    
    if status:
        print("Status:", status)

    # Шаг фазы
    phase_increment = 2.0 * np.pi * current_frequency / SAMPLE_RATE
    
    # Генерируем массив фаз
    phases = current_phase + np.arange(frames) * phase_increment
    
    # Генерируем синусоиду (обязательно float32)
    wave = (current_amplitude * np.sin(phases)).astype(np.float32)

    # Сохраняем конечную фазу
    current_phase = (current_phase + frames * phase_increment) % (2.0 * np.pi)
    
    # --- УЛЬТИМАТИВНЫЙ ТЕСТ: ШЛЕМ СИГНАЛ ВО ВСЕ КАНАЛЫ ---
    for ch in range(outdata.shape[1]):
        outdata[:, ch] = wave 

def update_freq(val):
    global current_frequency
    current_frequency = float(val)

def update_amp(val):
    global current_amplitude
    # Переводим проценты (0-100) в множитель (0.0-1.0)
    current_amplitude = float(val) / 100.0

def main():
    # Попробуем использовать DirectSound (ID 8) или WASAPI (ID 10)
    # Если снова будет ошибка, поменяйте DEVICE_ID на 10 или 20
    DEVICE_ID = 10
    
    # 1. Автоматически узнаем требования Windows к этому устройству
    device_info = sd.query_devices(DEVICE_ID)
    actual_sample_rate = int(device_info['default_samplerate'])
    max_channels = device_info['max_output_channels']
    
    print(f"--- Настройка аудио ---")
    print(f"Устройство ID: {DEVICE_ID} ({device_info['name']})")
    print(f"Требуемая частота ОС: {actual_sample_rate} Hz")
    print(f"Доступно каналов: {max_channels}")
    print(f"-----------------------")
    
    # Обновляем глобальную частоту для формулы генерации волны
    global SAMPLE_RATE
    SAMPLE_RATE = actual_sample_rate

    sd.default.device = DEVICE_ID

    # 2. Запускаем поток строго с теми параметрами, которые дала ОС
    # 2. Запускаем поток с монопольным доступом (Exclusive Mode)
    stream = sd.OutputStream(
        samplerate=actual_sample_rate, 
        channels=max_channels, 
        dtype='float32',
        callback=audio_callback,
        blocksize=1024,
        extra_settings=sd.WasapiSettings(exclusive=True)  # <--- ДОБАВЬТЕ ЭТО
    )
    stream.start()

    # --- Создаем простой графический интерфейс (GUI) ---
    root = tk.Tk()
    root.title("HSD Haptic Controller (Live)")
    root.geometry("400x200")

    tk.Label(root, text="Частота вибрации (Hz) [Рекомендуется: 40 - 150]").pack(pady=(10, 0))
    freq_slider = tk.Scale(
        root, from_=10, to=300, orient=tk.HORIZONTAL, 
        resolution=1, length=300, command=update_freq
    )
    freq_slider.set(current_frequency)
    freq_slider.pack()

    tk.Label(root, text="Сила вибрации (%)").pack(pady=(10, 0))
    amp_slider = tk.Scale(
        root, from_=0, to=100, orient=tk.HORIZONTAL, 
        resolution=1, length=300, command=update_amp
    )
    amp_slider.set(current_amplitude * 100)
    amp_slider.pack()

    print("Аудиопоток запущен. Закройте окно, чтобы остановить генерацию.")
    
    root.mainloop()

    stream.stop()
    stream.close()

if __name__ == "__main__":
    main()