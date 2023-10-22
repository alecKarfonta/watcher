import cv2
import time
from threading import Thread

from watcher import Watcher


# defining a helper class for implementing multi-threading 
class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id # default is 0 for main camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False 
        self.stopped = True
        # thread instantiation  
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t.start()
    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()
    # method to return latest read frame 
    def read(self):
        return self.frame
    # method to stop reading frames 
    def stop(self):
        self.stopped = True

import torch 

def main():
    print ("main")
    watcher = Watcher(pre_init=True)

    cam_1_string = 'rtsp://admin:@192.168.1.104'
    
    webcam_stream = WebcamStream(stream_id=cam_1_string) # 0 id for main camera
    webcam_stream.start()
    
    frame = webcam_stream.read()
    width = len(frame)
    height = len(frame[0])
    shape = frame.shape[:2]
    target_sizes = torch.tensor([shape])

    out = cv2.VideoWriter(
                'filename_1.avi',  
                cv2.VideoWriter_fourcc(*'MJPG'), 
                15, 
                shape
                ) 


    objects = watcher.get_objects(frame, target_sizes, is_profile=True)
    frame = watcher.overlay_boxes(objects, frame, width, height)

    num_frames_processed = 0 
    start = time.time()
    step = 0
    cap_step = 10
    try:
        while True :
            if step % 100 == 0:
                print (f"{step} frames")

            if webcam_stream.stopped is True :
                break
        
            frame = webcam_stream.read()

            frame = watcher.overlay_boxes(objects, frame, width, height)

            out.write(frame) 

            # adding a delay for simulating video processing time 
            delay = 0.03 # delay value in seconds
            time.sleep(delay) 
            if step % cap_step == 0:
                objects = watcher.get_objects(frame, target_sizes, is_profile=True)
            num_frames_processed += 1
            # displaying frame 
            #cv2.imshow('frame' , frame)
            #out.write(frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            step += 1
    except KeyboardInterrupt:
        out.release()
        webcam_stream.vcap.release()


    end = time.time()
    webcam_stream.stop() # stop the webcam stream
    elapsed = end-start
    
    fps = num_frames_processed/elapsed 
    print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))
    # closing all windows 
    cv2.destroyAllWindows()
    
    #stream_1 = VideoStreamWidget(cam_1_string, "Cam1")
    #stream_1.update()
    #stream_1.show_frame()

    #video_getter = VideoGet(cam_1_string).start()
    #video_shower = VideoShow(video_getter.frame).start()
    #cps = CountsPerSec().start()

    #while True:
    #    if video_getter.stopped or video_shower.stopped:
    #        video_shower.stop()
    #        video_getter.stop()
    #        break

    #    frame = video_getter.frame
    #    #frame = putIterationsPerSec(frame, cps.countsPerSec())
    #    video_shower.frame = frame
    #    #cps.increment()

#if __name__ == "__main__":
#    main()

import threading
import time
from queue import Queue
import cv2

frames = Queue(10)

class ImageGrabber(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.ID=ID
        self.cam=cv2.VideoCapture(ID)

    def run(self):
        global frames
        while True:
            ret, frame = self.cam.read()
            frames.put(frame)
            time.sleep(0.03)


class Main(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):

        watcher = Watcher(pre_init=True)

        out = cv2.VideoWriter(
                    'filename_1.avi',  
                    cv2.VideoWriter_fourcc(*'MJPG'), 
                    15, 
                    (1920, 1080)
                    ) 
        

        target_sizes = torch.tensor([(1080, 1920)])
        global frames
        step = 0
        frames_read = 0
        objects = []
        try:
            while True: #and step < 100:
                #if step % 100 == 0:
                #    print(f"{step} frames")
                step += 1
                # If has frames to process
                if(not frames.empty()):
                    # Pop a frame from the queue
                    self.Currframe = frames.get()


                    #frames.task_done()
                    frames_read += 1
                    if frames_read % 100 == 0:
                        print(f"{frames_read} frames in {step} steps")

                    
                    if frames_read % 8 == 0:
                        objects = watcher.get_objects(self.Currframe, target_sizes, is_profile=True)
                        if objects:
                            print (f"found {len(objects)} objects")
                        else:
                            print ("No objects found")
                    
                    if objects:
                        self.Currframe = watcher.overlay_boxes(objects, self.Currframe, 1080, 1920)

                    #print (f"Read frame: {self.Currframe[0] = }")
                    out.write(self.Currframe) 

                if frames_read > 1000:
                    print ("Done. Read 1000 frames.")
                    break
        except KeyboardInterrupt:
            out.release()

        out.release()
cam_1_string = 'rtsp://admin:Cody3.1415@192.168.1.104'
cam_2_string = 'rtsp://admin:admin@192.168.1.153'

grabber = ImageGrabber(cam_2_string)
main = Main()

grabber.start()
main.start()
main.join()
grabber.join()


