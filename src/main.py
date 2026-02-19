import sys
import os
from predict import predict_disease

def main():
    # 1. Check if the user provided an image path
    if len(sys.argv) < 2:
        print("No image provided.")
        print("Usage: python main.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    # 2. Check if the file actually exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find the file '{image_path}'")
        sys.exit(1)
        
    print(f"\nLoading image: {image_path}")
    
    # 3. Read the image as raw bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        
    # 4. Pass the bytes to the prediction 'engine'
    segmented_img, results = predict_disease(image_bytes)
    
    # 5. Print the formatted results
    for i, res in enumerate(results):
        disease_name = res['disease']
        confidence = res['confidence']
        
        # Add a star to the top prediction
        if i == 0:
            print(f"1. {disease_name}: {confidence}%")
        else:
            print(f"{i+1}. {disease_name}: {confidence}%")
    segmented_img.show()
    print("=====================================\n")

if __name__ == "__main__":
    main()