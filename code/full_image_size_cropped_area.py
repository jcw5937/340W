def process_image_and_extract_groups(image_path):
    # Define the regex pattern
    pattern = r'Mass-Test_P_(\d+)_(\w+)_(\w+)_'  # Fro testing image
    # pattern = r'Mass-Training_P_(\d+)_(\w+)_(\w+)_' # For training image
    match = re.search(pattern, image_path)
    if match:
        number = match.group(1)
        left = match.group(2)
        cc = match.group(3)
        result = f'Mass-Test_P_{number}_{left}_{cc}' # Update this part accordingly
        # result = f'Mass-Training_P_{number}_{left}_{cc}' # Update this part accordingly
        # Open the image
        image = Image.open(image_path)

        # Convert the grayscale image to a NumPy array
        image_array = np.array(image)

        # Calculate the non-zero area
        nonzero_pixels = np.count_nonzero(image_array)
        total_pixels = image_array.size
        nonzero_area_percentage = (nonzero_pixels / total_pixels) * 100

        return result, image, nonzero_area_percentage
    else:
        print(f"No match found for file: {image_path}")
        return None, None, None
