# Ячейка для тестирования модели YOLOv5 в Jupyter Notebook
import torch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def test_yolo_model(model_path="results/football_model/weights/best.pt",
                    images_folder="test_images",
                    output_folder="results/football_model/test_output",
                    show_results=True,
                    clear_output_folder=True):

    images_folder = Path(images_folder)
    output_folder = Path(output_folder)

    if clear_output_folder and output_folder.exists():
        for f in output_folder.glob("*"):
            if f.is_file():
                f.unlink()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Загружаем модель
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    for img_path in images_folder.glob("*.*"):  # JPG, PNG и т.д.
        results = model(str(img_path))

        # Получаем изображения с предсказаниями прямо в памяти
        rendered_imgs = results.render()  # список numpy изображений

        # Сохраняем результат
        for i, img_array in enumerate(rendered_imgs):
            save_path = output_folder / f"{img_path.stem}_pred{img_path.suffix}"
            # Конвертируем RGB обратно в BGR для cv2.imwrite
            cv2.imwrite(str(save_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            # Показываем результат в ноутбуке
            if show_results:
                plt.figure(figsize=(8,8))
                plt.imshow(img_array)
                plt.title(img_path.name)
                plt.axis('off')
                plt.show()

def test_yolo_video(model_path="results/football_model/weights/best.pt",
                    video_path="test_videos/test_video.mp4",
                    output_video="results/football_model/test_output/test_output_video.mp4",
                    conf_threshold=0.25,
                    show_frames=False,
                    max_frames=None):

    video_path = Path(video_path)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Загружаем модель
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.conf = conf_threshold  # порог уверенности

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        results = model(frame)
        img_result = results.render()[0]  # render() возвращает список изображений

        out.write(cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))

        if show_frames:
            plt.figure(figsize=(8,8))
            plt.imshow(img_result)
            plt.axis('off')
            plt.show()

    cap.release()
    out.release()
    print(f"✅ Видео с предсказаниями сохранено в {output_video}")                
