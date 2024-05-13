from identifysnowpack import identify_image

input_path = input("Enter the input image: ")
shape_path = input("Enter the shapefile: ")
output_path = input("Enter the output image: ")
identify_image(input_path, shape_path, output_path)