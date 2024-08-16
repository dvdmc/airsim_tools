import numpy as np

# A function that obtains the different colors of an image
def compute_class_colors(rgb_image, prev_class_colors=[]):
    """A function that obtains the different colors of an image as integers
    Returns:
        class_colors: a list of colors
    """
    class_colors = prev_class_colors
    rgb_np = np.array(rgb_image)
    for row in rgb_np:
        for pixel in row:
            if pixel.tolist() not in class_colors:
                class_colors.append(pixel.tolist())
    return class_colors

def get_color_map(dataset_name: str, bgr: bool = True) -> np.ndarray:
    """
        Get the color map for the dataset
        Args:
            dataset_name: name of the dataset
        Returns:
            color_map: color map for the dataset
    """
    if dataset_name == "coco_voc":
        color_map = get_pascal_labels(bgr=bgr)
    elif dataset_name == "pascal_8":
        color_map = get_pascal_8_labels(bgr=bgr)
    elif dataset_name == "nyu2":
        color_map = get_nyu2_40_labels(bgr=bgr)
    elif dataset_name == "nyu2_14":
        color_map = get_nyu2_14_classes(bgr=bgr)
    elif dataset_name == "airsim":
        color_map = get_airsim_labels(bgr=bgr)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return color_map

def get_pascal_labels(bgr=False):
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [0, 64, 0],  # 1=aeroplane # TEMPORAL CHANGE
            [0, 128, 0],  # 2=bicycle
            [128, 128, 0],  # 3=bird
            [0, 0, 128],  # 4=boat
            [128, 0, 128],  # 5=bottle
            [0, 128, 128],  # 6=bus
            [128, 128, 128],  # 7=car
            [64, 0, 0],  # 8=cat
            [192, 0, 0],  # 9=chair
            [64, 128, 0],  # 10=cow
            [192, 128, 0],  # 11=diningtable
            [64, 0, 128],  # 12=dog
            [192, 0, 128],  # 13=horse
            [64, 128, 128],  # 14=motorbike
            [192, 128, 128],  # 15=person
            [0, 64, 0],  # 16=potted plant
            [128, 64, 0],  # 17=sheep
            [0, 192, 0],  # 18=sofa
            [128, 192, 0],  # 19=train
            [0, 64, 128],  # 20=tv/monitor
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_pascal_labels_names():
    return [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

# Get pascal labels without background
def get_pascal_labels_wo_background(bgr=False):
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (20, 3)
    """
    color_map = np.array(
        [
            [0, 64, 0],  # 1=aeroplane # TEMPORAL CHANGE
            [0, 128, 0],  # 2=bicycle
            [128, 128, 0],  # 3=bird
            [0, 0, 128],  # 4=boat
            [128, 0, 128],  # 5=bottle
            [0, 128, 128],  # 6=bus
            [128, 128, 128],  # 7=car
            [64, 0, 0],  # 8=cat
            [192, 0, 0],  # 9=chair
            [64, 128, 0],  # 10=cow
            [192, 128, 0],  # 11=diningtable
            [64, 0, 128],  # 12=dog
            [192, 0, 128],  # 13=horse
            [64, 128, 128],  # 14=motorbike
            [192, 128, 128],  # 15=person
            [0, 64, 0],  # 16=potted plant
            [128, 64, 0],  # 17=sheep
            [0, 192, 0],  # 18=sofa
            [128, 192, 0],  # 19=train
            [0, 64, 128],  # 20=tv/monitor
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_pascal_8_labels(bgr=False):
    """Load the mapping that associates 7 pascal classes with label colors
    Returns:
        np.ndarray with dimensions (7, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [128, 0, 128],  # 5=bottle, purple
            [192, 0, 0],  # 9=chair, light red
            [192, 128, 0],  # 11=diningtable, light blue
            [192, 128, 128],  # 15=person, pink
            [0, 64, 0],  # 16=potted plant, dark green
            [0, 192, 0],  # 18=sofa, light green
            [0, 64, 128],  # 20=tv/monitor, dark blue
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map


def get_pascal_7_labels_wo_background(bgr=False):
    """Load the mapping that associates 7 pascal classes with label colors
    Returns:
        np.ndarray with dimensions (7, 3)
    """
    color_map = np.array(
        [
            [128, 0, 128],  # 5=bottle, purple
            [192, 0, 0],  # 9=chair, light red
            [192, 128, 0],  # 11=diningtable, light blue
            [192, 128, 128],  # 15=person, pink
            [0, 64, 0],  # 16=potted plant, dark green
            [0, 192, 0],  # 18=sofa, light green
            [0, 64, 128],  # 20=tv/monitor, dark blue
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_nyu2_40_labels(bgr=False):
    """Load the mapping that associates NYU2 classes with label colors
    Returns:
        np.ndarray with dimensions (41, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [174, 199, 232],  # 1=wall
            [152, 223, 138],  # 2=floor
            [31, 119, 180],  # 3=cabinet
            [255, 187, 120],  # 4=bed
            [188, 189, 34],  # 5=chair
            [140, 86, 75],  # 6=sofa
            [255, 152, 150],  # 7=table
            [214, 39, 40],  # 8=door
            [197, 176, 213],  # 9=window
            [148, 103, 189],  # 10=bookshelf
            [196, 156, 148],  # 11=picture
            [23, 190, 207],  # 12=counter
            [178, 76, 76],  # 13=blinds
            [247, 182, 210],  # 14=desk
            [66, 188, 102],  # 15=shelves
            [219, 219, 141],  # 16=curtain
            [140, 57, 197],  # 17=dresser
            [202, 185, 52],  # 18=pillow
            [51, 176, 203],  # 19=mirror
            [200, 54, 131],  # 20=floormat
            [92, 193, 61],  # 21=clothes
            [78, 71, 183],  # 22=ceiling
            [172, 114, 82],  # 23=books
            [255, 127, 14],  # 24=refrigerator
            [91, 163, 138],  # 25=television
            [153, 98, 156],  # 26=paper
            [140, 153, 101],  # 27=towel
            [158, 218, 229],  # 28=showercurtain
            [100, 125, 154],  # 29=box
            [178, 127, 135],  # 30=whiteboard
            [120, 185, 128],  # 31=person
            [146, 111, 194],  # 32=nightstand
            [44, 160, 44],  # 33=toilet
            [112, 128, 144],  # 34=sink
            [96, 207, 209],  # 35=lamp
            [227, 119, 194],  # 36=bathtub
            [213, 92, 176],  # 37=bag
            [94, 106, 211],  # 38=otherstructure
            [82, 84, 163],  # 39=otherfurniture
            [100, 85, 144],  # 40=otherprop
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_nyu2_39_labels_wo_background(bgr=False):
    """Load the mapping that associates NYU2 classes with label colors
    Returns:
        np.ndarray with dimensions (41, 3)
    """
    color_map = np.array(
        [
            [174, 199, 232],  # 1=wall
            [152, 223, 138],  # 2=floor
            [31, 119, 180],  # 3=cabinet
            [255, 187, 120],  # 4=bed
            [188, 189, 34],  # 5=chair
            [140, 86, 75],  # 6=sofa
            [255, 152, 150],  # 7=table
            [214, 39, 40],  # 8=door
            [197, 176, 213],  # 9=window
            [148, 103, 189],  # 10=bookshelf
            [196, 156, 148],  # 11=picture
            [23, 190, 207],  # 12=counter
            [178, 76, 76],  # 13=blinds
            [247, 182, 210],  # 14=desk
            [66, 188, 102],  # 15=shelves
            [219, 219, 141],  # 16=curtain
            [140, 57, 197],  # 17=dresser
            [202, 185, 52],  # 18=pillow
            [51, 176, 203],  # 19=mirror
            [200, 54, 131],  # 20=floormat
            [92, 193, 61],  # 21=clothes
            [78, 71, 183],  # 22=ceiling
            [172, 114, 82],  # 23=books
            [255, 127, 14],  # 24=refrigerator
            [91, 163, 138],  # 25=television
            [153, 98, 156],  # 26=paper
            [140, 153, 101],  # 27=towel
            [158, 218, 229],  # 28=showercurtain
            [100, 125, 154],  # 29=box
            [178, 127, 135],  # 30=whiteboard
            [120, 185, 128],  # 31=person
            [146, 111, 194],  # 32=nightstand
            [44, 160, 44],  # 33=toilet
            [112, 128, 144],  # 34=sink
            [96, 207, 209],  # 35=lamp
            [227, 119, 194],  # 36=bathtub
            [213, 92, 176],  # 37=bag
            [94, 106, 211],  # 38=otherstructure
            [82, 84, 163],  # 39=otherfurniture
            [100, 85, 144],  # 40=otherprop
        ]
    )

    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_nyu2_14_classes(bgr=False):
    """Load the mapping that associates NYU2 classes with label colors
    Returns:
        np.ndarray with dimensions (13, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [255, 187, 120],  # 1=bed
            [172, 114, 82],  # 2=books
            [78, 71, 183],  # 3=ceiling
            [188, 189, 34],  # 4=chair
            [152, 223, 138],  # 5=floor
            [140, 153, 101],  # 6=furniture
            [255, 127, 14],  # 7=objects
            [161, 171, 27],  # 8=picture
            [190, 225, 64],  # 9=sofa
            [206, 190, 59],  # 10=table
            [115, 176, 195],  # 11=tv
            [153, 108, 6],  # 12=wall
            [247, 182, 210],  # 13=window
        ]
    )

    if bgr:
        color_map = color_map[:, ::-1]
    return color_map


def get_airsim_labels(bgr=False):

    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [153, 108, 6],  # 1=aeroplane
            [112, 105, 191],  # 2=bicycle
            [89, 121, 72],  # 3=bird
            [190, 225, 64],  # 4=boat
            [206, 190, 59],  # 5=bottle
            [81, 13, 36],  # 6=bus
            [115, 176, 195],  # 7=car
            [161, 171, 27],  # 8=cat
            [135, 169, 180],  # 9=chair
            [29, 26, 199],  # 10=cow
            [102, 16, 239],  # 11=diningtable
            [242, 107, 146],  # 12=dog
            [156, 198, 23],  # 13=horse
            [49, 89, 160],  # 14=motorbike
            [68, 218, 116],  # 15=person
            [11, 236, 9],  # 16=potted plant
            [196, 30, 8],  # 17=sheep
            [121, 67, 28],  # 18=sofa
            [0, 53, 65],  # 19=train
            [146, 52, 70],  # 20=tv/monitor
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

def get_airsim_labels2(bgr=False):

    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [153, 108, 6],  # 5=bottle
            [112, 105, 191],  # 9=chair
            [89, 121, 72],  # 11=diningtable
            [116, 218, 68],  # 15=person
            [206, 190, 59],  # 16=potted plant
            [81, 13, 36],  # 18=sofa
            [115, 176, 195],  # 20=tv/monitor
        ]
    )
    if bgr:
        color_map = color_map[:, ::-1]
    return color_map

# Transform rgb colors to class labels
def rgb_to_class(rgb_image, class_colors):
    """A function that converts each color of an rgb image in a matrix with class labels (class_colors can be getPascalLabels())
    Returns:
        class_image: a matrix with the same shape as the input image, where each pixel is a class label
    """
    rgb_np = np.array(rgb_image, dtype=np.uint8)[:, :, :3]
    class_image = np.zeros(rgb_np.shape[:2], dtype=np.uint8)

    for class_label, class_color in enumerate(class_colors):
        mask = np.all(rgb_np == class_color, axis=-1)
        class_image[mask] = class_label
    return class_image

# Transform class labels to rgb colors
def class_to_rgb(class_image: np.ndarray, class_colors: np.ndarray) -> np.ndarray:
    """A function that converts each class label of a class image in a matrix with rgb colors (class_colors can be getPascalLabels())
    Returns:
        rgb_image: a matrix with the same shape as the input image, where each pixel is a rgb color
    """
    rgb_image = np.zeros(
        (class_image.shape[0], class_image.shape[1], 3), dtype=np.uint8)
    for class_label, class_color in enumerate(class_colors):
        mask = class_image == class_label
        rgb_image[mask] = class_color
    return rgb_image

def label2rgb(label_map, label_colors): # Not related to sent info, only display
    num_classes = label_colors.shape[0]
    label_colors = label_colors[:num_classes, :]
    rgb_label_map = class_to_rgb(label_map, label_colors)

    return rgb_label_map

def rgb2label(rgb_label_map, label_colors):
    num_classes = label_colors.shape[0]
    label_colors = label_colors[:num_classes, :]
    # Save img
    label_map = rgb_to_class(rgb_label_map, label_colors)
    return label_map



# Color map used in ADE20K dataset
ADE20K_COLOR_MAP = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], 
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], 
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], 
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], 
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], 
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], 
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], 
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], 
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], 
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], 
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], 
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255 ,82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], 
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], 
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], 
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255], 
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], 
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], 
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], 
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], 
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], 
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], 
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163], 
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235], 
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0], 
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], 
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255], 
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0], 
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]])

# Generic color map from COCO dataset
COCO_COLOR_MAP = np.array([
    [0, 0, 0], [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], 
    [106, 0, 228], [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], 
    [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], 
    [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], 
    [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], 
    [255, 179, 240], [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], 
    [255, 99, 164], [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], 
    [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], 
    [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95], 
    [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210], 
    [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0],
    [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], 
    [119, 0, 170], [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0], 
    [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185], [79, 210, 114], 
    [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45], 
    [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], 
    [191, 162, 208], [255, 255, 128], [147, 211, 203], [150, 100, 100], [168, 171, 172], 
    [146, 112, 198], [210, 170, 100], [92, 136, 89], [218, 88, 184], [241, 129, 0], 
    [217, 17, 255], [124, 74, 181], [70, 70, 70], [255, 228, 255], [154, 208, 0], 
    [193, 0, 92], [76, 91, 113], [255, 180, 195], [106, 154, 176], [230, 150, 140], 
    [60, 143, 255], [128, 64, 128], [92, 82, 55], [254, 212, 124], [73, 77, 174], 
    [255, 160, 98], [255, 255, 255], [104, 84, 109], [169, 164, 131], [225, 199, 255],
    [137, 54, 74], [135, 158, 223], [7, 246, 231], [107, 255, 200], [58, 41, 149], 
    [183, 121, 142], [255, 73, 97], [107, 142, 35], [190, 153, 153], [146, 139, 141], 
    [70, 130, 180], [134, 199, 156], [209, 226, 140], [96, 36, 108], [96, 96, 96], 
    [64, 170, 64], [152, 251, 152], [208, 229, 228], [206, 186, 171], [152, 161, 64], 
    [116, 112, 0], [0, 114, 143], [102, 102, 156], [250, 141, 255]])
# List of all AirSim colors
# [[0, 0, 0], [153, 108, 6], [112, 105, 191], [89, 121, 72], [190, 225, 64],
# [206, 190, 59], [81, 13, 36], [115, 176, 195], [161, 171, 27], [135, 169, 180],
# [29, 26, 199], [102, 16, 239], [242, 107, 146], [156, 198, 23], [49, 89, 160],
# [68, 218, 116], [11, 236, 9], [196, 30, 8], [121, 67, 28], [0, 53, 65],
# [146, 52, 70], [226, 149, 143], [151, 126, 171], [194, 39, 7], [205, 120, 161],
# [212, 51, 60], [211, 80, 208], [189, 135, 188], [54, 72, 205], [103, 252, 157],
# [124, 21, 123], [19, 132, 69], [195, 237, 132], [94, 253, 175], [182, 251, 87],
# [90, 162, 242], [199, 29, 1], [254, 12, 229], [35, 196, 244], [220, 163, 49],
# [86, 254, 214], [152, 3, 129], [92, 31, 106], [207, 229, 90], [125, 75, 48],
# [98, 55, 74], [126, 129, 238], [222, 153, 109], [85, 152, 34], [173, 69, 31],
# [37, 128, 125], [58, 19, 33], [134, 57, 119], [218, 124, 115], [120, 0, 200],
# [225, 131, 92], [246, 90, 16], [51, 155, 241], [202, 97, 155], [184, 145, 182],
# [96, 232, 44], [133, 244, 133], [180, 191, 29], [1, 222, 192], [99, 242, 104],
# [91, 168, 219], [65, 54, 217], [148, 66, 130], [203, 102, 204], [216, 78, 75],
# [234, 20, 250], [109, 206, 24], [164, 194, 17], [157, 23, 236], [158, 114, 88],
# [245, 22, 110], [67, 17, 35], [181, 213, 93], [170, 179, 42], [52, 187, 148],
# [247, 200, 111], [25, 62, 174], [100, 25, 240], [191, 195, 144], [252, 36, 67],
# [241, 77, 149], [237, 33, 141], [119, 230, 85], [28, 34, 108], [78, 98, 254],
# [114, 161, 30], [75, 50, 243], [66, 226, 253], [46, 104, 76], [8, 234, 216],
# [15, 241, 102], [93, 14, 71], [192, 255, 193], [253, 41, 164], [24, 175, 120],
# [185, 243, 231], [169, 233, 97], [243, 215, 145], [72, 137, 21], [160, 113, 101],
# [214, 92, 13], [167, 140, 147], [101, 109, 181], [53, 118, 126], [3, 177, 32],
# [40, 63, 99], [186, 139, 153], [88, 207, 100], [71, 146, 227], [236, 38, 187],
# [215, 4, 215], [18, 211, 66], [113, 49, 134], [47, 42, 63], [219, 103, 127],
# [57, 240, 137], [227, 133, 211], [145, 71, 201], [217, 173, 183], [250, 40, 113],
# [208, 125, 68], [224, 186, 249], [69, 148, 46], [239, 85, 20], [108, 116, 224],
# [56, 214, 26], [179, 147, 43], [48, 188, 172], [221, 83, 47], [155, 166, 218],
# [62, 217, 189], [198, 180, 122], [201, 144, 169], [132, 2, 14], [128, 189, 114],
# [163, 227, 112], [45, 157, 177], [64, 86, 142], [118, 193, 163], [14, 32, 79],
# [200, 45, 170], [74, 81, 2], [59, 37, 212], [73, 35, 225], [95, 224, 39],
# [84, 170, 220], [159, 58, 173], [17, 91, 237], [31, 95, 84], [34, 201, 248],
# [63, 73, 209], [129, 235, 107], [231, 115, 40], [36, 74, 95], [238, 228, 154],
# [61, 212, 54], [13, 94, 165], [141, 174, 0], [140, 167, 255], [117, 93, 91],
# [183, 10, 186], [165, 28, 61], [144, 238, 194], [12, 158, 41], [76, 110, 234],
# [150, 9, 121], [142, 1, 246], [230, 136, 198], [5, 60, 233], [232, 250, 80],
# [143, 112, 56], [187, 70, 156], [2, 185, 62], [138, 223, 226], [122, 183, 222],
# [166, 245, 3], [175, 6, 140], [240, 59, 210], [248, 44, 10], [83, 82, 52],
# [223, 248, 167], [87, 15, 150], [111, 178, 117], [197, 84, 22], [235, 208, 124],
# [9, 76, 45], [176, 24, 50], [154, 159, 251], [149, 111, 207], [168, 231, 15],
# [209, 247, 202], [80, 205, 152], [178, 221, 213], [27, 8, 38], [244, 117, 51],
# [107, 68, 190], [23, 199, 139], [171, 88, 168], [136, 202, 58], [6, 46, 86],
# [105, 127, 176], [174, 249, 197], [172, 172, 138], [228, 142, 81], [7, 204, 185],
# [22, 61, 247], [233, 100, 78], [127, 65, 105], [33, 87, 158], [139, 156, 252],
# [42, 7, 136], [20, 99, 179], [79, 150, 223], [131, 182, 184], [110, 123, 37],
# [60, 138, 96], [210, 96, 94], [123, 48, 18], [137, 197, 162], [188, 18, 5],
# [39, 219, 151], [204, 143, 135], [249, 79, 73], [77, 64, 178], [41, 246, 77],
# [16, 154, 4], [116, 134, 19], [4, 122, 235], [177, 106, 230], [21, 119, 12],
# [104, 5, 98], [50, 130, 53], [30, 192, 25], [26, 165, 166], [10, 160, 82],
# [106, 43, 131], [44, 216, 103], [255, 101, 221], [32, 151, 196], [213, 220, 89],
# [70, 209, 228], [97, 184, 83], [82, 239, 232], [251, 164, 128], [193, 11, 245],
# [38, 27, 159], [229, 141, 203], [130, 56, 55], [147, 210, 11], [162, 203, 118], [255, 255, 255]]