// stb single-header implementations (image decode/encode) - replaces OpenCV
// imgcodecs so the portable build avoids the GDAL/DB/poppler dependency closure.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
