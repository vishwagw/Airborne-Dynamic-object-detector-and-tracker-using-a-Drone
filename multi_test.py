# Replace the single video path with a list
vid_paths = ['military_jet.mp4', 'video2.mp4', 'video3.mp4']  # Add your video files here

for vid_path in vid_paths:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Failed to open video: {vid_path}")
        continue
    
    # Define output path based on input video name
    output_path = f"output_{vid_path.split('.')[0]}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Processing: {vid_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Reset tracker for each video
    tracker = SimpleTracker()
    prev_positions = {}
    prev_times = {}
    
    # Rest of the processing loop remains the same
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video: {vid_path}")
            break
        # ... (rest of the loop as in the modified script) ...
    
    cap.release()
    out.release()
    print(f"Output video saved to: {output_path}")

cv2.destroyAllWindows()
print("System shutdown complete.")