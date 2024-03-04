from pycocotools import mask as coco_mask
from detectron2.structures import BoxMode


def create_coco_format(data_pairs):
    file_dicts = []

    for i, pair in enumerate(data_pairs):
        
        filename = pair[0]

        _, _, _, height, width, _ = pair[1][0].split()
        #print(i)
        img_item = {}
        img_item['file_name'] = filename
        img_item['image_id'] = i
        img_item['height']= int(height)
        img_item['width']= int(width)

        objs = []
        for line in pair[1]:
            # print('Line: ', line)
            _, object_id, _, height, width, rle = line.split()

            #Obtain instance_id
            class_id = int(object_id) // 1000

            if class_id == 10:
                continue
            elif class_id == 1:
                class_id = 2
            elif class_id == 2:
                class_id = 0

            # Create LRE to match COCO’s compressed RLE format 
            seg = {"size": [int(height), int(width)], "counts": rle}
            
            #Extract BB
            bbox = coco_mask.toBbox(seg)
            
            obj = {
                "bbox": list(map(float,bbox)),
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": seg,
                "category_id": class_id,
            }
            objs.append(obj)
        img_item['annotations'] = objs
        file_dicts.append(img_item)

    return file_dicts   