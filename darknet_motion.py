from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue

import numpy as np
import imutils
from collections import deque

# Input to Step 5: Helper function
# Calculate the foreground frame based on frames
def get_movement(frames, shape, q):
    movement_frame = np.zeros(shape, dtype='float32')
    i = 0
    for f in frames:
        i += 1
        movement_frame += f * i
    movement_frame = movement_frame / ((1 + i) / 2 * i)
    movement_frame[movement_frame > 254] = 255
    q.put(movement_frame)
    #return movement_frame


# Input to Step 5: Helper function
# Calculate the background frame based on frames
# This function has obvious improvement potential
# - Could avoid to recalculate full list every time
def get_background(frames, shape):
    bg = np.zeros(shape, dtype='float32')
    for frame in frames:
        bg += frame
    bg /= len(frames)
    bg[bg > 254] = 255
    return bg


# Detect and return boxes of moving parts
def detect(frame, bg_frames, fg_frames, threshold=15, min_box=200):
    # Step 3-4: Add the frame to the our list of foreground and background frames
    fg_frames.append(frame)
    bg_frames.append(frame)

    # Input to Step 5: Calculate the foreground and background frame based on the lists
    #fg_frame = get_movement(list(fg_frames), frame.shape)
    q = Queue()
    t = Thread(target=get_movement,args=(list(fg_frames), frame.shape,q))
    t.start()
    bg_frame = get_background(list(bg_frames), frame.shape)
    fg_frame = q.get()
    t.join()
    

    # Step 5: Calculate the difference to detect movement
    movement = cv2.absdiff(fg_frame, bg_frame)
    movement[movement < threshold] = 0
    movement[movement > 0] = 254
    movement = movement.astype('uint8')
    movement = cv2.cvtColor(movement, cv2.COLOR_BGR2GRAY)
    movement[movement > 0] = 254
    # As we don't return the movement frame, we show it here for debug purposes
    # Should be removed before release
    #cv2.imshow('Movement', movement)

    # Step 6: Find the list of contours
    contours = cv2.findContours(movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Step 7: Convert them to boxes
    boxes = []
    for contour in contours:
        # Ignore small boxes
        if cv2.contourArea(contour) < min_box:
            continue
        # Convert the contour to a box and append it to the list
        box = cv2.boundingRect(contour)
        boxes.append(box)

    return boxes

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def yolov4_detect(queue, thresh):            
    last_frame_no = -1
    last_detected = -1
    global interval    
    file_no = 0    
    random.seed(3)  # deterministic bbox colors        
    video = set_saved_video(cap, 'object_'+args.out_filename, (video_width, video_height))            

    while True:        
        frame_no = queue.get()
        if frame_no == -1:
            break      

        frame = queue.get()  

        if last_frame_no > frame_no: #reset varilables when switching to a new video stream.
            last_frame_no = -1
            last_detected = -1
            interval = 0                           
        
        if interval >= video_fps // 2:
            interval = video_fps // 2  # upper limit of frame detection interval .             
              
        if (frame_no-last_frame_no) >= interval: #detect one frame every other [interval] frames.                               
            last_frame_no = frame_no                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                    interpolation=cv2.INTER_LINEAR)        
            img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
            darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())                             
            
            detections = darknet.detect_image(network, class_names, img_for_detect, thresh)            

            detections_adjusted = []            
            if len(detections) > 0 : #target object detected                                  
                last_detected = frame_no
                for label, confidence, bbox in detections:
                    bbox_adjusted = convert2original(frame, bbox)
                    detections_adjusted.append((str(label), confidence, bbox_adjusted))
                image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
                cv2.imwrite('motions/' + str(file_no)+'.jpg',image)         
                file_no += 1                    
                interval = 5 #As soon as target object is detected, set detection interval to the lowset.    
            elif (frame_no-last_detected) == interval: #no target object detected, but last detection was successful.           
                    cv2.imwrite('motions/' + str(file_no)+'.jpg',frame)  
                    file_no += 1
            else:                                     
                    interval += 0.20 #Both last and the latest detecctions failed, increase detection interval to decrease chances of false alarms.

        if args.out_filename is not None:
            if (last_detected == frame_no): # write video when target object is detected in current frame.
                video.write(image)                
            elif (frame_no-last_detected) <= interval: # write video when current frame is within half of a second from last detected frame.
                video.write(frame)   
                    
    
    video.release()  
    print('yolov4 output video closed.')      

def main_thread(raw_queue,width=928, height=480, scale_factor=3):
    quit = False
    frame_no = 0
    last_time = time.time()

    while not quit:
        ret, frame = cap.read()
        if not ret:                        
            break

        # Resize the frame
        frame = cv2.resize(frame, (width, height))
        # Step 1: Scale down to improve speed (only takes integer scale factors)
        work_frame = cv2.resize(frame, (width // scale_factor, height // scale_factor))
        # Step 2: Blur it and convert the frame to float32
        work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
        work_frame_f32 = work_frame.astype('float32')             

        raw_queue.put(frame_no)
        raw_queue.put(frame) #raw frame
        raw_queue.put(work_frame_f32) #cv2 processed frame

        if frame_no < raw_queue.qsize():
            fps = int(frame_no / (time.time() - last_time))       
        else:
            fps = int((frame_no-raw_queue.qsize()) / (time.time() - last_time))       

        if (frame_no % video_fps) == 0:
            text = "FPS:" + str(fps)            

        frame_no += 1 

        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(frame, 'read_out:'+str(raw_queue.qsize()), (260, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(frame, 'motion_detect:'+str(frame_queue.qsize()), (560, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)      

        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit = True              

        while raw_queue.qsize() >= 300:
            time.sleep(1/1000)            

        frame_no += 1
    
    return quit

def motion_detect(raw_queue, queue):
    # Create the buffer of our lists
    bg_frames = deque(maxlen=30)
    fg_frames = deque(maxlen=10)

    # Get the webcam
    #cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)    

    # We want to see how many frames per second we process    
    
    frame_cnt = 0    
    record_cnt = 0
    recording = False
    file_no = 0    
    skip = False
    quit = False    
    global interval
    min_box = 49        
    
    while not quit:
        # Step 0: Read the webcam frame (ignore return code)        
        frame_no = raw_queue.get()
        if frame_no == -1:            
            break

        frame = raw_queue.get()
        work_frame_f32 = raw_queue.get()
             
        if interval < 5:
            min_box = 49
        else:
            min_box = int((interval+2)**2)        
        
        # Step 3-7 (steps in function): Detect all the boxes around the moving parts
        boxes = detect(work_frame_f32, bg_frames, fg_frames, 15, min_box)             
        if len(boxes) > 0:            
            if skip:
                frame_cnt += 1
                if (frame_cnt % 5) == 0:
                    skip = False
                    frame_cnt = 0
            else:
                #cv2.imwrite('motions/' + filename + str(file_no)+'.jpg',frame)            
                video_out.write(frame)
                #file_no += 1
                skip = True

            recording = True
            
        if recording:                  
            queue.put(frame_no) 
            queue.put(frame)                                
            record_cnt += 1            
            if record_cnt >= video_fps:
                record_cnt = 0
                recording = False

    return quit

if __name__ == '__main__':
    
    raw_queue = Queue()
    frame_queue = Queue()    

    args = parser()    
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    interval = 0
    first_time = True    
    scale_factor = 3

    files = os.listdir(input_path)
    files.sort()    

    for f in files:       
#        filename = f.replace('192.168.1.99_','')                
#        filename = filename.replace('.h264','-')                        
        cap = cv2.VideoCapture(input_path+f)       
        video_fps =  int(cap.get(cv2.CAP_PROP_FPS))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print('processing',input_path+f,str(video_width)+'x'+str(video_height)+'@'+str(video_fps)+'FPS')

        if first_time:        
            print('threh:',args.thresh)       
            t1 = Thread(target=motion_detect, args=(raw_queue, frame_queue))
            t2 = Thread(target=yolov4_detect, args=(frame_queue, args.thresh))
            t1.start()
            t2.start()
            first_time = False
            video_out = set_saved_video(cap, 'motion_'+args.out_filename, (video_width, video_height))

        quit = main_thread(raw_queue,video_width, video_height, scale_factor)
        cap.release()        

        if quit: # quit flag represents that user has pressed key 'q' in motion_detect() function.
            break        

    while raw_queue.qsize() > 0 or frame_queue.qsize() > 0:
        time.sleep(1/1000)

    raw_queue.put(-1)
    frame_queue.put(-1)
    
    if len(files) > 0:
        video_out.release()
        print('motion output video closed.')      
        t1.join()
        t2.join()
    else:
        print('source folder is emtpy.')

    print('All threads joined.')      
    cv2.destroyAllWindows()

    print('program exit.')
