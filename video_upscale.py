import cv2
import torch
import numpy as np
import time  # Import the time module

def upscale_video(input_path, output_path, model, device, scale_factor=2):
    start_time = time.time()  # Record the start time

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2, 0, 1))
        frame = np.ascontiguousarray(frame)

        input_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_frame = output_tensor.squeeze().cpu().numpy()
        output_frame = output_frame.transpose((1, 2, 0))
        output_frame = (output_frame * 255.0).clip(0, 255).astype(np.uint8)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        out.write(output_frame)

    cap.release()
    out.release()

    end_time = time.time()  # Record the end time
    time_taken = end_time - start_time  # Calculate the time taken

    print(f"Time taken to upscale the video: {time_taken:.2f} seconds")
