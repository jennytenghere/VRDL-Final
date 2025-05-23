import os
import random
import pandas as pd

def generate_validation_data(image_folder, output_csv):
    if not os.path.exists(image_folder):
        print(f"Not exits {image_folder}")
        return

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    if len(image_files) == 0:
        print("No image")
        return

    validation_data = []
    for image in image_files:
        fold = random.randint(0, 4)
        validation_data.append({"image_id": image, "fold": fold})

    df = pd.DataFrame(validation_data)

    df.to_csv(output_csv, index=False)
    print(f"Generate {output_csv}, {len(image_files)} images")

if __name__ == "__main__":

    image_folder = "/home/chiehniteng/Desktop/vrdl/final/input/cassava-leaf-disease-classification/train_images"
    output_csv = "validation_data.csv"

    generate_validation_data(image_folder, output_csv)
