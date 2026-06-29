import struct
import cv2


def convert_to_custom_binary(image_path, output_path):
    # 1. Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # 2. Convert to grayscale
    # Even though it's grayscale, we will retain the structural metadata
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Extract metadata
    # For a grayscale image, channels = 1
    height, width, channels = rgb_img.shape

    # Stride (bytes per row) = width * channels * bytes_per_pixel
    # Since it's uint8 (1 byte per pixel) and 1 channel, stride == width
    stride = width * channels

    # 4. Prepare the binary header
    # 'i' represents a 4-byte signed integer (int32)
    # We pack: width, height, channels, stride
    header = struct.pack("iiii", width, height, channels, stride)

    # 5. Get the raw uint8 data bytes
    data_bytes = rgb_img.tobytes()

    # 6. Write header and data to the binary file
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(data_bytes)

    print(f"Successfully converted {image_path} to {output_path}")
    print(f"Metadata written -> Width: {width}, Height: {height}, Channels: {channels}, Stride: {stride}")


# Example usage:
if __name__ == "__main__":
    # Replace with your image path and desired output path
    input_image = "input.jpg"
    output_bin = "output.bin"

    # try:
    convert_to_custom_binary(input_image, output_bin)
    # except Exception as e:
    # print(f"Error: {e}")
