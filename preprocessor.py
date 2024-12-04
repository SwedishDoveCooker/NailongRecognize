import os
import glob
from PIL import Image, ImageSequence

def process_gif(gif_path, output_dir):
    try:
        gif = Image.open(gif_path)
        first_frame = next(ImageSequence.Iterator(gif))
        first_frame = first_frame.convert('RGB')
        frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(gif_path))[0]}.jpg")
        first_frame.save(frame_path, 'JPEG')
        print(f"Saved first frame to {frame_path}")
        
        gif.close()
        os.remove(gif_path)
        print(f"Deleted original GIF file: {gif_path}")
    except PermissionError as e:
        print(f"Failed to delete {gif_path}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {gif_path}: {e}")

def convert_images_to_jpg(directory):
    image_files = glob.glob(os.path.join(directory, '*.*'))
    
    for image_file in image_files:
        if image_file.lower().endswith('.gif'):
            if directory == './input':
                continue
            else:
                process_gif(image_file, directory)
        elif not image_file.lower().endswith(('.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov')):
            img = Image.open(image_file).convert('RGB')
            jpg_path = os.path.join(directory, f"{os.path.splitext(os.path.basename(image_file))[0]}.jpg")
            img.save(jpg_path, 'JPEG')
            print(f"Converted {image_file} to {jpg_path}")
            os.remove(image_file)
            print(f"Deleted original file: {image_file}")
