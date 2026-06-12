import os
import cv2
import sys
import argparse

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper
import numpy as np


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("crack")

coco_id_list = [1]


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def dfl_decode(box_data):
    """Distribution Focal Loss decode for YOLOv11"""
    # box_data: (4, N*16) after reshape from (N, 4*16)
    # Returns: (N, 4) decoded box coordinates (ltrb format)
    import torch
    x = torch.tensor(box_data).float()
    n, c = x.shape
    p_num = 4
    mc = c // p_num
    
    # Reshape: (N, 4, 16) -> (N*4, 16)
    x = x.reshape(n, p_num, mc)
    
    # Softmax over the 16 bins
    x = torch.softmax(x, dim=-1)
    
    # Weight sum: (0, 1, 2, ..., 15) weighted by probabilities
    weight = torch.arange(mc).float().reshape(1, 1, mc)
    decoded = (x * weight).sum(dim=-1)  # (N, 4)
    
    return decoded.numpy()

def post_process(input_data):
    # YOLOv11 ultralytics format: (1, 4+1+num_classes, 8400)
    # 37 = 4(box) + 1(conf) + 32(classes)
    output = input_data[0]  # (1, 37, 8400)
    
    # 三个尺度: 80x80, 40x40, 20x20
    scales = [80*80, 40*40, 20*20]
    
    boxes, scores, classes_conf = [], [], []
    
    # 解耦输出: [4(框), 1(置信度), 32(类别)]
    box_output = output[:, :4, :]      # (1, 4, 8400)
    conf_output = output[:, 4:5, :]    # (1, 1, 8400)
    cls_output = output[:, 5:, :]      # (1, 32, 8400)
    
    # 逐尺度处理
    box_idx = 0
    for scale in scales:
        boxes_scale = box_output[:, :, box_idx:box_idx+scale]  # (1, 4, N)
        conf_scale = conf_output[:, :, box_idx:box_idx+scale]  # (1, 1, N)
        cls_scale = cls_output[:, :, box_idx:box_idx+scale]    # (1, 32, N)
        box_idx += scale
        
        # 计算网格大小
        grid_size = int(np.sqrt(scale))
        grid_h = grid_w = grid_size
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(grid_h, grid_w)
        row = row.reshape(grid_h, grid_w)
        grid = np.stack([col, row], axis=-1).reshape(-1, 2)  # (N, 2)
        
        stride_w = IMG_SIZE[0] // grid_w
        stride_h = IMG_SIZE[1] // grid_h
        
        # reshape for processing: (1, 4, N) -> (N, 4)
        boxes_s = boxes_scale[0].T  # (N, 4)
        conf_s = conf_scale[0].T    # (N, 1)
        cls_s = cls_scale[0].T      # (N, 32)
        
        # YOLOv11 解码公式 (ultralytics 风格):
        # cx = (grid_x + sigmoid(tx)) * stride
        # cy = (grid_y + sigmoid(ty)) * stride
        # w = pw * exp(tw) = sigmoid(bw) * stride (使用 exp 变换)
        # h = ph * exp(th) = sigmoid(bh) * stride
        
        # 应用 sigmoid 到 xy，exp 到 wh
        bx = 1 / (1 + np.exp(-boxes_s[:, 0]))
        by = 1 / (1 + np.exp(-boxes_s[:, 1]))
        bw = np.exp(boxes_s[:, 2])  # exp 变换
        bh = np.exp(boxes_s[:, 3])  # exp 变换
        
        # 计算绝对坐标
        cx = (grid[:, 0] + bx) * stride_w
        cy = (grid[:, 1] + by) * stride_h
        w = bw * stride_w
        h = bh * stride_h
        
        # 转换为 xyxy 格式
        boxes_xyxy = np.zeros((len(cx), 4), dtype=np.float32)
        boxes_xyxy[:, 0] = cx - w * 0.5  # left
        boxes_xyxy[:, 1] = cy - h * 0.5  # top
        boxes_xyxy[:, 2] = cx + w * 0.5  # right
        boxes_xyxy[:, 3] = cy + h * 0.5  # bottom
        
        # 应用 sigmoid 到 conf 和 cls
        conf_s = 1 / (1 + np.exp(-conf_s))
        cls_s = 1 / (1 + np.exp(-cls_s))
        
        # 计算类别得分
        cls_max = np.max(cls_s, axis=-1)
        classes = np.argmax(cls_s, axis=-1)
        scores_scale = cls_max * conf_s.flatten()
        
        # 获取类别置信度
        cls_conf = cls_s[np.arange(len(classes)), classes]
        
        # 过滤低置信度
        mask = scores_scale > OBJ_THRESH
        boxes.append(boxes_xyxy[mask])
        classes_conf.append(cls_conf[mask])
        scores.append(scores_scale[mask])
    
    # 三个尺度: 80x80, 40x40, 20x20
    scales = [80*80, 40*40, 20*20]
    
    boxes, scores, classes_conf = [], [], []
    
    # 解耦输出: [4(框), 1(置信度), 32(类别)]
    box_output = output[:, :4, :]      # (1, 4, 8400)
    conf_output = output[:, 4:5, :]    # (1, 1, 8400)
    cls_output = output[:, 5:, :]      # (1, 32, 8400)
    
    # 逐尺度处理
    box_idx = 0
    for scale in scales:
        boxes_scale = box_output[:, :, box_idx:box_idx+scale]  # (1, 4, N)
        conf_scale = conf_output[:, :, box_idx:box_idx+scale]  # (1, 1, N)
        cls_scale = cls_output[:, :, box_idx:box_idx+scale]    # (1, 32, N)
        box_idx += scale
        
        # 计算网格大小
        grid_size = int(np.sqrt(scale))
        grid_h = grid_w = grid_size
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(grid_h, grid_w)
        row = row.reshape(grid_h, grid_w)
        grid = np.stack([col, row], axis=-1).reshape(-1, 2)  # (N, 2)
        
        stride_w = IMG_SIZE[0] // grid_w
        stride_h = IMG_SIZE[1] // grid_h
        
        # reshape for processing: (1, 4, N) -> (N, 4)
        boxes_s = boxes_scale[0].T  # (N, 4) - directly take first batch and transpose
        conf_s = conf_scale[0].T    # (N, 1)
        cls_s = cls_scale[0].T      # (N, 32)
        
        # 直接坐标解码
        boxes_xywh = boxes_s  # (N, 4) - 假设是直接的 xywh 值
        
        # xywh decode: 转换为 xyxy
        # grid 是左上角坐标，加上 0.5 偏移
        cx = grid[:, 0] + 0.5 - boxes_xywh[:, 0]
        cy = grid[:, 1] + 0.5 - boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        
        # Convert to xyxy and apply stride
        boxes_xyxy = np.zeros((len(cx), 4), dtype=np.float32)
        boxes_xyxy[:, 0] = (cx - w * 0.5) * stride_w  # left
        boxes_xyxy[:, 1] = (cy - h * 0.5) * stride_h  # top
        boxes_xyxy[:, 2] = (cx + w * 0.5) * stride_w  # right
        boxes_xyxy[:, 3] = (cy + h * 0.5) * stride_h  # bottom
        
        # 应用 sigmoid
        conf_s = 1 / (1 + np.exp(-conf_s))
        cls_s = 1 / (1 + np.exp(-cls_s))
        
        # 计算类别得分
        cls_max = np.max(cls_s, axis=-1)
        classes = np.argmax(cls_s, axis=-1)
        scores_scale = cls_max * conf_s.flatten()
        
        # 获取类别置信度
        cls_conf = cls_s[np.arange(len(classes)), classes]
        
        # 过滤低置信度
        mask = scores_scale > OBJ_THRESH
        boxes.append(boxes_xyxy[mask])
        classes_conf.append(cls_conf[mask])
        scores.append(scores_scale[mask])
    
    if len(boxes) == 0 or all(len(b) == 0 for b in boxes):
        return None, None, None
    
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    
    # filter
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf.reshape(-1, 1))
    
    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
    
    if not nclasses and not nscores:
        return None, None, None
    
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    
    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # data params
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    # coco val folder: '../../../datasets/COCO//val2017'
    parser.add_argument('--img_folder', type=str, default='../model', help='img folder path')
    parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    co_helper = COCO_test_helper(enable_letter_box=True)

    # run test
    for i in range(len(img_list)):
        print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        '''
        # using for test input dumped by C.demo
        img_src = np.fromfile('./input_b/demo_c_input_hwc_rgb.txt', dtype=np.uint8).reshape(640,640,3)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
        '''

        # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
        pad_color = (0,0,0)
        img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocee if not rknn model
        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2,0,1))
            input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
            input_data = input_data/255.
        else:
            input_data = img

        outputs = model.run([input_data])
        boxes, classes, scores = post_process(outputs)

        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))
            img_p = img_src.copy()
            if boxes is not None:
                draw(img_p, co_helper.get_real_box(boxes), scores, classes)

            if args.img_save:
                if not os.path.exists('./result'):
                    os.mkdir('./result')
                result_path = os.path.join('./result', img_name)
                cv2.imwrite(result_path, img_p)
                print('Detection result save to {}'.format(result_path))
                        
            if args.img_show:
                cv2.imshow("full post process result", img_p)
                cv2.waitKeyEx(0)

        # record maps
        if args.coco_map_test is True:
            if boxes is not None:
                for i in range(boxes.shape[0]):
                    co_helper.add_single_record(image_id = int(img_name.split('.')[0]),
                                                category_id = coco_id_list[int(classes[i])],
                                                bbox = boxes[i],
                                                score = round(scores[i], 5).item()
                                                )

    # calculate maps
    if args.coco_map_test is True:
        pred_json = args.model_path.split('.')[-2]+ '_{}'.format(platform) +'.json'
        pred_json = pred_json.split('/')[-1]
        pred_json = os.path.join('./', pred_json)
        co_helper.export_to_json(pred_json)

        from py_utils.coco_utils import coco_eval_with_json
        coco_eval_with_json(args.anno_json, pred_json)

    # release
    model.release()
